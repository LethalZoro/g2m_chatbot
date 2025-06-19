
import asyncio
import json
import os
from typing import AsyncGenerator
import re


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
# ==============================================================================
# 1. DEFINE CUSTOM AGENTS FOR EACH DATA SOURCE
# ==============================================================================
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")



class SimpleRouter(BaseAgent):
    """
    Correctly extracts the user's query from ctx.user_content and places
    it into the 'r_user_query' key in the agent's state.
    """
    name: str = "SimpleRouter"
    description: str = "Extracts user input from the context and routes it for processing."
    output_key: str = "r_user_query"

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        initial_query = ""

        # We found the user's input in ctx.user_content.
        # It's an object with a 'parts' list, and the text is in the first part.
        if hasattr(ctx, 'user_content') and ctx.user_content and ctx.user_content.parts:
            initial_query = ctx.user_content.parts[0].text

        if not initial_query:
            # This is a fallback in case the input is ever empty.
            error_msg = "FATAL: Router could not find the user's query text in ctx.user_content."
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"error_message_router": error_msg})
            )
            return

        # Success! Silently update the state so your other agents can use the query.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={self.output_key: initial_query}
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
            
            # retriever_object = {
            #     "text_similarity_reranker": {
            #         "retriever": {
            #             "standard": {
            #                 "query": {
            #                     "semantic": {
            #                         "field": "text",
            #                         "query": query
            #                     }
            #                 }
            #             }
            #         },
            #         "field": "text",  # Field to use for reranking
            #         "inference_text": query  # The original query text for reranking context
            #     }
            # }
            # search_response = client.search(index="ask-g2m-serverless", retriever=retriever_object, size=5)

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

# ==============================================================================
# 2. CREATE THE MULTI-AGENT WORKFLOW (MODIFIED)
# ==============================================================================

# query_router = LlmAgent(
#     name="QueryRouter",
#     model="gemini-2.0-flash-001", 
#     instruction="""You are a silent router. The user will provide a question.
#                  Your only task is to extract the user's question and place it
#                  into the 'r_user_query' state variable without generating any other output.""",
#     output_key="r_user_query"
# )

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
    ]
)

answer_synthesizer = LlmAgent(
    name="AnswerSynthesizer",
    model="gemini-2.5-flash-preview-05-20", 
    # model="gemini-2.5-pro-preview-06-05",
    instruction=(

        "You are an expert research analyst. Your ONLY goal is to answer the user's question by creating a detailed response that can be a table or normal paragraph as well using the information provided in the data sources below.\n\n"
        "**CRITICAL INSTRUCTIONS:**\n"
        "1.  Carefully analyze the text in 'Vector Search Results' and 'Semantic Search Results' to find the data needed to answer the user's question: '{state[r_user_query]}'\n"
        "2.  You MUST construct a Markdown table when requested by the user.\n"
        "3.  Answer using ONLY the data found in the provided search results. Look for tables, lists, and descriptive text in the content.\n"
        "4.  For any specific data points or columns you cannot find information for in the provided text, you MUST write 'Information Not Found' in that specific cell. Do not leave any cells blank.\n"
        "5.  **DO NOT apologize or refuse to answer.** Your job is to build the most complete table possible with the information you are given, even if it is partial.\n"
        "6.  At the end of your response, create a '## Sources Used' section and list the file names of the documents you used to generate the response. Cite the exact file name.\n\n"
        "**Data from Sources:**\n"
        "--- Vector Search Results ---\n{state[pinecone_data]}\n\n"
        "--- Semantic Search Results ---\n{state[elastic_data]}"
    )
)

root_agent = SequentialAgent(
    name="g2m_agent",
    sub_agents=[
        SimpleRouter(),
        data_gatherer,
        answer_synthesizer,
    ]
)

# app = reasoning_engines.AdkApp(
#     agent=root_agent,
#     enable_tracing=True,
# )


# remote_app= agent_engines.create(
#     display_name="G2M Agent v1.3",
#     agent_engine=app,
#     requirements=["google-adk==1.2.1", "google-cloud-aiplatform==1.95.1", "google-cloud-storage==2.19.0", "openai==1.82.1", "pinecone==6.0.2", "elasticsearch==9.0.2", "python-dotenv"],
#     extra_packages=["./g2m_agent"]
# )
