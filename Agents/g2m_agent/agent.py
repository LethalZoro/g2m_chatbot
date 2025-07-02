
import asyncio
import json
import time
import os
from typing import AsyncGenerator
import re
from typing import AsyncGenerator, ClassVar
from google.cloud import bigquery
from vertexai.language_models import TextGenerationModel
from vertexai.generative_models import GenerativeModel,  Tool, Part
from google.cloud.aiplatform_v1beta1 import Tool as GapicTool
from datetime import date, datetime

import vertexai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from google.adk.agents import (BaseAgent, LlmAgent, ParallelAgent,
                               SequentialAgent)
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from openai import OpenAI
from pinecone import Pinecone

from vertexai.preview import reasoning_engines
from vertexai import agent_engines

from google.cloud import bigquery
from google.cloud import storage
from google.api_core.exceptions import NotFound
# Load environment variables from a .env file
load_dotenv()

from google.cloud import secretmanager

def get_secret(secret_id: str, version_id: str = "latest") -> str:
    """
    Retrieves a secret's value from Google Cloud Secret Manager.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    # Decode the payload AND strip any leading/trailing whitespace or newlines
    return response.payload.data.decode("UTF-8").strip()


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

class ContextualizerAgent(BaseAgent):
    """
    An agent that processes conversational history to transform a follow-up
    question into a complete, standalone query for downstream agents.
    """
    name: str = "ContextualizerAgent"
    description: str = "Manages chat history and creates a standalone query from conversational context."
    output_key: str = "r_user_query"

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        # 1. Get the new user query and the chat history from the state
        new_user_query = ""
        if hasattr(ctx, 'user_content') and ctx.user_content and ctx.user_content.parts:
            new_user_query = ctx.user_content.parts[0].text
        
        if not new_user_query:
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"error_message_context": "FATAL: Could not find the user's query text."})
            )
            return

        chat_history = ctx.session.state.get("chat_history", [])
        
        # 2. If there is no history, the first question is the standalone query.
        if not chat_history:
            print(f"[{self.name}] First question. Using query as is: '{new_user_query}'")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(
                    state_delta={self.output_key: new_user_query}
                )
            )
            return

        # 3. If there is history, build a prompt to create a standalone question
        # Format the chat history for the prompt
        formatted_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history])

        rewrite_prompt = f"""Given the conversation history below, rewrite the "Follow-up Question" into a complete, standalone question that can be understood without needing to read the history.

                **CRITICAL RULES:**
                1.  Your output MUST be ONLY the standalone question itself. Do not add any introductory text like "Here is the standalone question:".
                2.  If the "Follow-up Question" is already a complete question, simply return it unchanged.

                **Conversation History:**
                ---
                {formatted_history}
                ---

                **Follow-up Question:**
                "{new_user_query}"

                **Standalone Question:**
                """
        standalone_query = ""
        try:
            # 4. Call the LLM to perform the rewrite
            model = GenerativeModel("gemini-2.5-flash")
            response = await model.generate_content_async(rewrite_prompt)
            standalone_query = response.text.strip()
            print(f"[{self.name}] Original query: '{new_user_query}'")
            print(f"[{self.name}] Rewritten query: '{standalone_query}'")

        except Exception as e:
            # Fallback in case of an error during rewrite
            print(f"[{self.name}] WARN: Could not rewrite query due to an error: {e}. Falling back to original query.")
            standalone_query = new_user_query

        # 5. Yield the new standalone query to the state for the next agent
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={self.output_key: standalone_query}
            )
        )

class PineconeSearcher(BaseAgent):
    name: str = "PineconeSearcher"
    description: str = "Searches a Pinecone vector DB and formats results with sources."
    output_key: str | None = None

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        query = ctx.session.state.get("r_user_query")
        if not query:
            yield Event(author=self.name,
            actions=EventActions(
                state_delta={"error_message_pinecone": "Error: No user_query found in state."}
            ))
            return

        # print(f"[{self.name}] Embedding and searching for: {query.strip()}")
        formatted_results = ""
        try:
            pc = Pinecone(api_key= get_secret("PINECONE_API_KEY"))
            openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
            index = pc.Index("ask-g2m")
            embedding_response = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
            query_vector = embedding_response.data[0].embedding
            
            search_response = index.query(
                vector=query_vector,
                top_k=15,
                rerank={
                    "model": "bge-reranker-v2-m3",
                    "top_n": 5,
                    "rank_fields": ["text"]
                },
                fields=["category", "text"],
                include_metadata=True
            )
            
            hits = search_response.to_dict().get('matches', [])
            # print(f"Found {len(hits)} hits in Pinecone.")
            # print("Pinecone data: ",hits)
            if not hits:
                formatted_results = "No relevant documents found in Pinecone."
            else:
                for hit in hits:
                    doc_id = hit.get('id', 'Unknown ID')
                    score = hit.get('score', 0)
                    metadata = hit.get('metadata', {})
                    text_content = metadata.get('text', 'No text content.')

                    if text_content != 'No text content.':
                        text_content = text_content.strip()  # Remove leading/trailing whitespace
                        # Normalize multiple newlines: replace 3 or more newlines with 2 (one blank line)
                        text_content = re.sub(r'\n', ' ', text_content)

                    # Extract additional metadata for a richer source description
                    source_info_parts = [f"ID: {doc_id}"]
                    
                    file_name_from_meta = metadata.get('file_name')
                    if file_name_from_meta:
                        source_info_parts.append(f"File: {file_name_from_meta}")
                    
                    page_number_from_meta = metadata.get('page_number')
                    if page_number_from_meta:
                        source_info_parts.append(f"Page: {page_number_from_meta}")
                    
                    source_info_parts.append(f"Score: {score:.2f}")
                    
                    source_description = ", ".join(source_info_parts)
                    formatted_results += f"Source: Pinecone ({source_description})\nContent: {text_content}\n\n"
                    # print(f"source_description Pinecone: {source_description}")
            # print("Pinecone data: ",formatted_results)
        except Exception as e:
            formatted_results = f"An error occurred during Pinecone search: {e}"

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={self.output_key: formatted_results}
            )
        )

class ElasticSearcher(BaseAgent):
    name: str = "ElasticSearcher"
    description: str = "Performs semantic search on Elasticsearch and formats results with sources."
    output_key: str | None = None

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        query = ctx.session.state.get("r_user_query")
        if not query:
            yield Event(author=self.name,
            actions=EventActions(
                state_delta={"error_message_elastic": "Error: No user_query found in state."}
            ))
            return

        # print(f"[{self.name}] Searching for: {query.strip()}")
        formatted_results = ""
        try:
            client = Elasticsearch(
                "https://ask-g2m-c7a0bf.es.us-east-1.aws.elastic.cloud:443",
                api_key=get_secret("ELASTIC_API_KEY"),
            )
            retriever_object = {"standard": {"query": {"semantic": {"field": "text", "query": query}}}}
            search_response = client.search(index="ask-g2m-serverless", retriever=retriever_object, size=5)

            hits = search_response.get('hits', {}).get('hits', [])
            # print(f"Found {len(hits)} hits in Elasticsearch.")
            # print("Elasticsearch data: ",hits)
            if not hits:
                formatted_results = "No relevant documents found in Elasticsearch."
            else:
                for hit in hits:
                    source_data = hit.get('_source', {})
                    file_name = source_data.get('file_name', 'Unknown Document')
                    page_number = source_data.get('page_number', 'N/A')
                    text_content = source_data.get('text', 'No text content.')
                    if text_content != 'No text content.':
                        text_content = text_content.strip()
                        text_content = re.sub(r'\n', ' ', text_content) 

                    score = hit.get('_score', 0)
                    formatted_results += f"Source: {file_name} (Page: {page_number}, Score: {score:.2f})\nContent: {text_content}\n\n"
                    # print(f"source_description Elastic: {file_name} (Page: {page_number}, Score: {score:.2f})")
            # print("Elastic Search: ",formatted_results)   

        except Exception as e:
            formatted_results = f"An error occurred during Elasticsearch search: {e}"

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={self.output_key: formatted_results}
            )
        )


class IntelligentBigQueryAgent(BaseAgent):
    name: str = "IntelligentBigQueryAgent"
    description: str = "Dynamically generates and executes BigQuery SQL queries based on user questions and live schema data."
    output_key: str | None = None

    # --- Static Configuration ---
    DATASETS_TO_SCAN: ClassVar[list[str]] = [
                "tableau"
    ]

    # --- GCS Cache Configuration ---
    # IMPORTANT: REPLACE 'your-gcs-bucket-for-caching' WITH YOUR ACTUAL BUCKET NAME.
    CACHE_BUCKET_NAME: ClassVar[str] = "ask-g2m-table-bucket"
    CACHE_OBJECT_NAME: ClassVar[str] = "agent-cache/bigquery_schema.json"
    CACHE_EXPIRATION_SECONDS: ClassVar[int] = 86400  # Cache for 1 day (86400 seconds)
    # --- End Configuration ---

    async def _get_cached_schema(self, project_id: str) -> str:
        """
        Manages GCS-based caching for the BigQuery schema.
        It checks for a fresh cache object in GCS before deciding to fetch a new one.
        """
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.CACHE_BUCKET_NAME)
            blob = bucket.blob(self.CACHE_OBJECT_NAME)

            blob.reload()  # Get the latest metadata from GCS

            # Check if the cache object exists and is younger than the expiration time
            if blob.updated and ((time.time() - blob.updated.timestamp()) < self.CACHE_EXPIRATION_SECONDS):
                print("--- Returning schema from GCS CACHE ---")
                return blob.download_as_text()
        except NotFound:
            # This is expected if the cache doesn't exist yet. We'll create it below.
            print("--- Cache object not found. A new one will be created. ---")
        except Exception as e:
            # Log other potential errors but proceed to fetch a fresh schema
            print(f"Warning: Could not read from GCS cache. Will fetch fresh schema. Error: {e}")

        # If cache is expired or doesn't exist, fetch a new one and upload it
        print("--- Fetching new schema from BigQuery and uploading to GCS cache... ---")
        new_schema = await self._fetch_schema_from_bigquery(project_id)
        if not new_schema.startswith("Error:"):
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.CACHE_BUCKET_NAME)
                blob = bucket.blob(self.CACHE_OBJECT_NAME)
                blob.upload_from_string(new_schema, content_type="application/json")
                print("--- Successfully updated GCS cache. ---")
            except Exception as e:
                print(f"Warning: Failed to upload new schema to GCS cache. Error: {e}")
        return new_schema

    async def _fetch_schema_from_bigquery(self, project_id: str) -> str:
        """
        Connects to BigQuery and builds the schema string.
        (This is your original _get_schema_representation logic).
        """
        schema_representation = ""
        try:
            client = bigquery.Client(project=project_id)
            for dataset_id in self.DATASETS_TO_SCAN:
                schema_representation += f"Dataset: `{dataset_id}`\n"
                tables = client.list_tables(f"{project_id}.{dataset_id}")
                for table_ref in tables:
                    table = client.get_table(table_ref)
                    schema_representation += f"  Table: `{table.table_id}`\n"
                    if table.description:
                        schema_representation += f"    Description: {table.description}\n"
                    columns_str = ", ".join([f"{col.name} ({col.field_type})" for col in table.schema])
                    schema_representation += f"    Columns: [ {columns_str} ]\n"
                schema_representation += "\n"
        except Exception as e:
            if "Not found" in str(e):
                return f"Error: A dataset or table was not found. Please verify the names in DATASETS_TO_SCAN and check your agent's IAM permissions. Details: {e}"
            return f"Error: Could not retrieve schema from BigQuery. Details: {e}"
        return schema_representation

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        global PROJECT_ID
        user_query = ctx.session.state.get("r_user_query")
        if not user_query:
            yield Event(author=self.name,
                        actions=EventActions(
                            state_delta={"error_message_bigquery": "Error: No user_query found in state."}
                        ))
            return

        formatted_results = ""
        try:
            # 1. Get the schema representation (now using the caching layer)
            schema_info = await self._get_cached_schema(project_id=PROJECT_ID)
            if "Error:" in schema_info:
                yield Event(author=self.name, actions=EventActions(state_delta={self.output_key: schema_info}))
                return

            # 2. Use Vertex AI Gemini to generate the SQL query
            model = GenerativeModel("gemini-2.5-pro")
            text_to_sql_prompt = f"""You are an expert Google BigQuery data analyst. Your sole purpose is to convert a user's natural language question into a single, valid, and efficient BigQuery SQL query.

                **CRITICAL RULES:**
                1.  **Always use fully qualified table names:** All table names must be wrapped in backticks and include the project and dataset, like `g2m-dev.tableau.commodity_dir_hubspoke_join`.
                2.  **Prioritize Cleaned `g2m_` Columns:** When available, always prefer columns prefixed with `g2m_` (e.g., `g2m_buyer_name`, `g2m_buyer_type`) over their raw counterparts (e.g., `customer_name`) for the most accurate and standardized results.
                3.  **Use `purchase_amount` for Spend:** For any questions about total spend, revenue, or sales, you MUST use the `purchase_amount` column. Only if it's not suitable should you calculate spend by multiplying `order_quantity` by `unit_price`.
                4.  **Implement Flexible and Robust Filtering:** For any `WHERE` clause that filters on a string value from the user's question (like a buyer name, seller name, or product), you MUST make the search flexible. Achieve this by:
                    a. Wrapping the column name in `LOWER()` and `TRIM()` (e.g., `LOWER(TRIM(t1.g2m_buyer_name))`).
                    b. Using the `LIKE` operator.
                    c. Converting the user's search term to lowercase and enclosing it in `%` wildcards (e.g., `LIKE '%city of dallas%'`).
                5.  **No Explanations:** Your response must be ONLY the raw SQL query. Do not add any introductory text, closing remarks, or `sql` code blocks.

                **SCHEMA INFORMATION:**
                *Use the detailed guidance below first. Refer to the auto-generated schema only for a complete list of all available columns.*

                ---

                **1. Auto-Generated Schema Overview:**

                {schema_info}
                ---
                **2. Detailed Table Guidance & Key Concepts:**

                #### **Table: `g2m-dev.tableau.commodity_dir_hubspoke_join`**
                * **Purpose:** This table contains historical IT purchase transactions from the Texas DIR program. It links buyers, sellers, products, and contract details.

                * **Buyer (Who Purchased):**
                    * `g2m_buyer_name`: **Primary field for the buyer's name.** Use this for filtering by agency, city, or school district.
                    * `g2m_buyer_acronym`: Use to match common acronyms (e.g., 'TDCJ', 'UT').
                    * `g2m_buyer_type`: Buyer's vertical (e.g., `State`, `Higher Ed`, `K-12`).
                    * `customer_city`, `customer_state`: Buyer's location.

                * **Seller (Who Sold):**
                    * `who_sold`: **Use this for general "who sold it" questions.** It represents the seller on the PO, which could be a vendor or a reseller.
                    * `g2m_contract_broker_seller_name`: The primary contract holder with DIR.
                    * `g2m_vendor_agent_seller_name`: The reseller or partner who fulfilled the order.
                    * `g2m_supplier_vendor_seller_name`: **The brand or OEM of the product sold.** (e.g., 'Okta', 'Dell', 'Microsoft'). Use this to answer questions about specific products or brands.

                * **Product & Service Classification (What Was Bought):**
                    * **Use these fields to answer questions about categories of products.**
                    * `Market_Domain`: Broadest category (e.g., `Cybersecurity`, `Data Management`, `Infrastructure`).
                    * `Functional_Segment`: More specific functional area (e.g., `Data Integration`, `Cloud Database Management Systems`, `AI-Driven ITSM`).
                    * `Standard_Evaluation_Label`: A granular, analyst-assigned label (e.g., `Zero Trust`, `Digital Asset Management Platforms`).

              _details * `purchase_amount`: **Primary field for the total value of the transaction.**
                    * `order_date`: The specific date of the purchase.
.
                * **Diversity/HUB Status (Special Seller Attribute):**
                    * Use `vendor_hub_type` or `reseller_hub_type` to find minority, woman, or veteran-owned businesses (HUB certified).

                * **Contract Context:**
                    * `rfo_number`, `rfo_description`: Use to find purchases related to a specific solicitation.
                    * `contract_number`: The specific DIR contract identifier.
                ---
                **3. High-Quality Query Examples:**

                * **User Question:** "What cybersecurity solutions has the City of Dallas purchased in the past two years?"
                    * **Logic:** Filter by `LOWER(TRIM(g2m_buyer_name))` LIKE `'%city of dallas%'`, `Market_Domain` (or `Functional_Segment`), and `fiscal_year`.
                * **User Question:** "Which identity vendors are found in the Texas Higher Education market?"
                    * **Logic:** Filter by `Functional_Segment` ('Identity & Access Mgmt') and `g2m_buyer_type` ('Higher Ed'), then show `g2m_supplier_vendor_seller_name`.
                * **User Question:** "What is the total spend by K-12 on cloud infrastructure?"
                    * **Logic:** Filter by `g2m_buyer_type` ('K-12') and `Functional_Segment` ('Cloud'), then SUM `purchase_amount`.
                * **User Question:** "Which vendors fall under the ‘Zero Trust’ evaluation label?"
                     * **Logic:** Filter by `Standard_Evaluation_Label` ('Zero Trust') and show `g2m_contract_broker_seller_name` or `who_sold`.
                ---

                **User's Question:** "{user_query}"

                **BigQuery SQL Query:**
                """
            # text_to_sql_prompt = f"""You are an expert Google BigQuery data analyst. Your sole purpose is to convert a user's natural language question into a single, valid, and efficient BigQuery SQL query.

            #         **CRITICAL RULES:**
            #         1.  **Always use fully qualified table names:** All table names must be wrapped in backticks and include the project and dataset, like `g2m-dev.tableau.commodity_dir_hubspoke_join`.
            #         2.  **Prioritize Cleaned `g2m_` Columns:** When available, always prefer columns prefixed with `g2m_` (e.g., `g2m_buyer_name`) over their raw counterparts (e.g., `customer_name`) for the most accurate and standardized results.
            #         3.  **Perform Necessary Calculations:** If a user asks for total spend or amount, you must calculate it by multiplying `order_quantity` by `unit_price` if a `purchase_amount` field isn't directly suitable.
            #         4.  **No Explanations:** Your response must be ONLY the raw SQL query. Do not add any introductory text, closing remarks, or `sql` code blocks.

            #         **SCHEMA INFORMATION:**
            #         *Use the detailed guidance when available, and refer to the auto-generated schema for a complete list of tables and columns.*
            #         ---
            #         **1. Auto-Generated Schema Overview:**
            #         {schema_info}
            #         ---
            #         **2. Detailed Table Guidance:**

            #         #### **Table: `g2m-dev.tableau.commodity_dir_hubspoke_join`**
            #         * **Purpose:** This table contains historical IT purchase transactions from the Texas DIR program. It links buyers, sellers, products, and contract details.
            #         * **Key Concepts & Columns:**
            #             * **Buyer (Who Purchased):**
            #                 * `g2m_buyer_name`: **Primary field for buyer name.**
            #                 * `g2m_buyer_type`: Buyer's vertical (e.g., `State`, `Higher Ed`, `K-12`).
            #                 * `customer_city`, `customer_state`: Buyer's location.
            #             * **Seller (Who Sold):**
            #                 * `who_sold`: **Primary field for the seller's name.** This consolidates vendors and resellers.
            #                 * `vendor_name`: The primary contract holder.
            #                 * `reseller_name`: The partner who fulfilled the order.
            #             * **Product (What Was Bought):**
            #                 * `brand_name`: The specific product or brand purchased (e.g., `Okta`, `Dell`, 'Cloudfare').
            #             * **Transaction Details (How Much & When):**
            #                 * `purchase_amount`: The total value of the transaction. Use this for spend analysis.
            #                 * `order_quantity`, `unit_price`: Can be multiplied to get the total amount.
            #                 * `order_date`: The specific date of the purchase.
            #                 * `fiscal_year`: The fiscal year of the purchase.
            #             * **Diversity/HUB Status (Special Seller Attribute):**
            #                 * `vendor_hub_type`, `reseller_hub_type`: Use these to identify minority, woman, or veteran-owned businesses (HUB certified).
            #             * **Contract Context:**
            #                 * `rfo_number`, `rfo_description`: Use to find purchases related to a specific solicitation.
            #                 * `contract_number`: The specific DIR contract identifier.
            #         ---

            #         **User's Question:** "{user_query}"

            #         **BigQuery SQL Query:**
            #         """
            response = await model.generate_content_async(
                text_to_sql_prompt,
                generation_config={"temperature": 0.1, "top_p": 0.8}
            )
            generated_sql = response.text.strip()
            print(f"[BigQuery] Generated SQL: {generated_sql}")

            # 3. Application-level security guardrail
            FORBIDDEN_KEYWORDS = ["UPDATE", "DELETE", "INSERT", "DROP", "CREATE", "ALTER", "TRUNCATE"]
            if any(keyword in generated_sql.upper().split() for keyword in FORBIDDEN_KEYWORDS):
                formatted_results = "Error: The generated query contained a restricted keyword and was blocked by the application's security guardrails."
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={self.output_key: formatted_results})
                )
                return

            if generated_sql.startswith("```sql"):
                generated_sql = generated_sql[6:].strip("`").strip()

            if generated_sql.startswith("Error:") or not generated_sql:
                formatted_results = generated_sql
            else:
                # 4. Execute Query
                client = bigquery.Client(project=PROJECT_ID)
                query_job = client.query(generated_sql)
                rows = query_job.result()

                if rows.total_rows == 0:
                    formatted_results = "Query executed successfully but returned no results."
                else:
                    # 5. Process and Clean the results before passing them on
                    results_list = [dict(row) for row in rows]
                    
                    print(f"--- Cleaning and structuring {len(results_list)} rows from BigQuery ---")
                    
                    cleaned_results = []
                    for row in results_list:
                        cleaned_row = {}
                        for key, value in row.items():
                            if isinstance(value, float):
                                # Round floats to 2 decimal places for cleaner output
                                cleaned_row[key] = round(value, 2)
                            elif value is None:
                                # Replace None/null with a consistent "N/A" string
                                cleaned_row[key] = "N/A"
                            elif isinstance(value, (date, datetime)):
                                # Convert date/datetime objects to ISO string format
                                cleaned_row[key] = value.isoformat()
                            else:
                                # Keep all other values as they are
                                cleaned_row[key] = value
                        cleaned_results.append(cleaned_row)
                    print(f"--- Cleaned and structured {cleaned_results} rows from BigQuery ---")
                    # Now, format the CLEANED and STRUCTURED data to be passed to the next agent
                    formatted_results = json.dumps(cleaned_results, indent=2)
                    print("BigQuery data: ", formatted_results)

        except Exception as e:
            formatted_results = f"An error occurred during BigQuery processing: {e}"

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={self.output_key: formatted_results})
        )


class AnswerSynthesizerAgent(BaseAgent):
    """
    An agent that dynamically constructs a prompt using data from the state
    and synthesizes a final answer using an LLM.
    """
    name: str = "AnswerSynthesizerAgent"
    description: str = "Synthesizes a final answer from gathered data."
    # This agent will output the final response to the user, so no output_key is needed.
    output_key: str | None = None

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        
        # 1. Retrieve the user query and data from the session state
        user_query = ctx.session.state.get("r_user_query", "No user query found in state.")
        bigquery_data = ctx.session.state.get("bigquery_data", "No BigQuery data was found in state.")
        vector_search_results = ctx.session.state.get("pinecone_data", "No Vector Search Results found in state.")
        semantic_search_results = ctx.session.state.get("elastic_data", "No Semantic Search Results found in state.")
        print("session state: ", ctx.session.state)
        # 2. Dynamically construct the full prompt with the retrieved data
        synthesis_prompt = f"""
                You are an expert research analyst. Your ONLY goal is to answer the user's question by creating a detailed response, which can be a table or a paragraph, using the information provided in the data sources below.

                **CRITICAL INSTRUCTIONS:**
                1.  **Prioritize Internal Knowledge First:** You must answer the user's question, '{user_query}', by following this strict order of data source priority:
                    * **1st Priority:** `BigQuery Results`
                    * **2nd Priority:** `Vector Search Results` (including Semantic, Pinecone, Elastic, etc.)
                    * **Last Resort:** `Web Search`

                2.  **Answer from the Knowledge Stack:** First, analyze only the `BigQuery Results` and `Vector Search Results`. Synthesize a complete answer based *exclusively* on this internal data.

                3.  **Use Web Search as a Fallback:** After providing an answer from the internal knowledge stack, evaluate if it fully and correctly answers the user's question. If the answer is incomplete or if no internal data is available, you MUST then use your web search tool to find the most current and relevant information.

                4.  **Clearly Separate Answers:** If you use a web search, you must present that information as a separate part of the answer. For example, create a new heading like "## Additional Information from Web Search" to distinguish it from the answer derived from the internal data. Do not mix web data with internal data in the initial answer.

                5.  **Handle Tables from BigQuery:** If you create a table from the `BigQuery Results`, you MUST use the exact keys from the JSON objects as the column headers. Do not invent new column names.

                6.  **Cite All Sources:** If you use information from a web search, you MUST cite your sources by providing the URL at the end of your answer.

                7.  **Create a 'Sources Used' Section:** At the end of your entire response, list all sources you used, including the specific BigQuery tables queried, internal file names, and any website URLs.

                **Data from Sources:**
                ---
                --- Vector Search Results ---
                NONE

                --- Semantic Search Results ---
                NONE

                --- BigQuery Results ---
                {bigquery_data}
                ---
                """
        
                # --- BigQuery Results ---
                # NONE
                # """
                # --- Vector Search Results ---
                # {vector_search_results}

                # --- Semantic Search Results ---
                # {semantic_search_results}


        # 3. Call the generative model with the complete, data-filled prompt
        final_answer_text = ""
        try:
            search_tool = Tool._from_gapic(
                raw_tool=GapicTool(
                    google_search=GapicTool.GoogleSearch(),
                ),
            )
            model = GenerativeModel("gemini-2.5-pro"
                                    ,tools=[search_tool]
                                    )
            response = await model.generate_content_async(synthesis_prompt)
            final_answer_text = response.text
        except Exception as e:
            final_answer_text = f"An error occurred during the final answer synthesis: {e}"

        chat_history = ctx.session.state.get("chat_history", [])
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "model", "content": final_answer_text})


        # 4. Yield the final answer to the user
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={
                    self.output_key: final_answer_text,
                    "chat_history": chat_history # Persist the updated history
                }
                
            ),
            content={"parts": [{"text": final_answer_text}], "role": "model"}
        )



vertexai.init(
    project="g2m-dev",
    location="us-central1",
    staging_bucket="gs://ask-g2m-agent",
)

data_gatherer = ParallelAgent(
    name="ConcurrentDataFetcher",
    sub_agents=[
        PineconeSearcher(output_key="pinecone_data"),
        ElasticSearcher(output_key="elastic_data"),
        IntelligentBigQueryAgent(output_key="bigquery_data"),
    ]
)

# finalanswer_synthesizer = LlmAgent(
#     name="FinalAnswerSynthesizer",
#     model="gemini-2.5-pro",
#     instruction=(
#         "You are an expert research analyst. Your task is to create a final answer based on the data provided in the state.\n\n"
#         "User's Question: 'state[r_user_query]'\n\n"
#         "Final Answer:\n 'state[final_answer]'\n\n"
#         "Please synthesize a final answer that is clear, concise, and directly addresses the user's question."
#     )
# )

# MODIFIED: The AnswerSynthesizer is now aware of the BigQuery results
# answer_synthesizer = LlmAgent(
#     name="AnswerSynthesizer",
#     # model="gemini-2.5-flash-preview-05-20",
#     model="gemini-2.5-pro",
#     instruction=(
#         "You are an expert research analyst. Your task is to answer the user's question: '{state[r_user_query]}'\n\n"
#         "**CRITICAL RULES:**\n"
#         "1. **ONLY use the exact JSON data provided in the BigQuery Results section below**\n"
#         "2. **The column headers in your table MUST be the exact JSON keys from the data**\n"
#         "3. **Do NOT invent, assume, or hallucinate any column names**\n"
#         "4. **If creating a table, use ONLY the keys that exist in the JSON objects**\n"
#         "5. **Present the data exactly as provided - do not transform or interpret it**\n\n"
#         "**Example:** If the JSON contains:\n"
#         "```json\n"
#         "{'name': 'John', 'age': 30}\n"
#         "```\n"
#         "Your table headers must be 'name' and 'age' - nothing else.\n\n"
#         "**Data Sources:**\n"
#         "--- BigQuery Results ---\n"
#         "{state[bigquery_data]}\n\n"
#         "Create a properly formatted table using ONLY the keys present in the JSON data above."
#     )
    # instruction=(
    #     "You are an expert research analyst. Your ONLY goal is to answer the user's question by creating a detailed response, which can be a table or a paragraph, using the information provided in the data sources below.\n\n"
    #     "**CRITICAL INSTRUCTIONS:**\n"
    #     "1.  Carefully analyze the information in 'Vector Search Results', 'Semantic Search Results', and 'BigQuery Results' to find the data needed to answer the user's question: '{state[r_user_query]}'\n"
    #     "2.  **Your response MUST be derived exclusively from the data provided in the 'Data from Sources' section. Do not use any prior knowledge about what a table's columns should be.**\n"
    #     "3.  When creating a table from 'BigQuery Results', you **MUST** use the keys from the JSON objects as the column headers. You **MUST NOT** invent, hallucinate, or assume column names. The provided JSON is the absolute and only source of truth.\n"
    #     "4.  For any specific data points or columns you cannot find information for in other sources, you MUST write 'Information Not Found' in that specific cell. Do not leave any cells blank.\n"
    #     "5.  **DO NOT apologize or refuse to answer.** Your job is to build the most complete answer possible with the information you are given, even if it is partial.\n"
    #     "6.  At the end of your response, create a '## Sources Used' section. List the specific sources you used, such as file names or the BigQuery tables that were queried.\n\n"
    #     "**Data from Sources:**\n"
    #     "--- Vector Search Results ---\nNONE\n\n"
    #     "--- Semantic Search Results ---\nNONE\n\n"
    #     # "--- Vector Search Results ---\n{state[pinecone_data]}\n\n"
    #     # "--- Semantic Search Results ---\n{state[elastic_data]}\n\n"
    #     "--- BigQuery Results ---\n{state[bigquery_data]}"
    # )
# )

root_agent = SequentialAgent(
    name="g2m_agent",
    sub_agents=[
        ContextualizerAgent(),
        data_gatherer,
        # answer_synthesizer,
        AnswerSynthesizerAgent(output_key="final_answer"),
        # finalanswer_synthesizer,
    ]
)

app = reasoning_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
)


remote_app= agent_engines.create(
    display_name="G2M Agent v1.6 Big Quer with web search",
    agent_engine=app,
    requirements=["google-adk==1.4.2", "google-cloud-aiplatform==1.98.0", "google-cloud-storage==2.19.0", "openai==1.82.1", "pinecone==6.0.2", "elasticsearch==9.0.2", "python-dotenv","google-cloud==0.34.0"],
    extra_packages=["./g2m_agent"]
)


# how much Okta did Texas Comptroller of Public Accounts buy in the last three years?
