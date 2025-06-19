import os
import sys
from typing import List, Dict, Any
import json
import uvicorn
import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
import uuid


# --- FastAPI App Initialization ---
# Initialize the FastAPI app with metadata for the documentation
app = FastAPI(
    title="Vertex AI Agent Interaction API",
    description="An API to list deployments, create sessions, and interact with existing Vertex AI Agents.",
    version="1.0.0",
)

# --- Vertex AI Initialization ---
# Load environment variables from a .env file for local development
load_dotenv()

# Get project configuration from environment variables
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
BUCKET = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")

# A single startup event to initialize Vertex AI and check for required variables
@app.on_event("startup")
def startup_event():
    """Initializes Vertex AI SDK on application startup."""
    if not all([PROJECT_ID, LOCATION, BUCKET]):
        print(
            "FATAL ERROR: Missing required environment variables: "
            "GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STAGING_BUCKET"
        )
        # In a real app, you might want a more graceful shutdown
        sys.exit(1)

    print("Initializing Vertex AI...")
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET,
    )
    print("Vertex AI Initialized Successfully.")

# --- Pydantic Models for Request & Response ---
# These models define the expected structure of your API's JSON data.
# FastAPI uses them for validation, serialization, and documentation.

class DeploymentResponse(BaseModel):
    resource_name: str = Field(..., example="projects/your-project/locations/us-central1/agentEngines/12345")

class SessionRequest(BaseModel):
    user_id: str = Field(..., example="test-user-123")
    resource_id: str = Field(..., example="123456689")

class SessionResponse(BaseModel):
    session_id: str = Field(..., example="4768534100908703744")
    user_id: str = Field(..., example="a123")
    resource_id: str = Field(..., example="3976792820177436672")
    last_update_time: str = Field(..., example="1750166696.884852")

class MessageRequest(BaseModel):
    user_id: str = Field(..., example="abc1")
    resource_id: str = Field(..., example="3976792820177436672")
    session_id: str = Field(..., example="3217044029279567872")
    msg_req: str = Field(..., example="Hi how are you")

class MessageResponse(BaseModel):
    response: List[Dict[str, Any]] = Field(..., example=[{"chunk": {"text": "hru today?"}}])


# --- Agent Core Functions ---
# These functions contain the core logic for interacting with the Vertex AI Agent Engine.


def list_all_deployments() -> List[Dict[str, str]]:
    """Lists all deployments."""
    deployments = agent_engines.list()
    return [{"resource_name": dep.resource_name} for dep in deployments]


def create_new_session(resource_id: str, user_id: str) -> Dict[str, Any]:
    """Creates a new session for a user."""
    remote_app = agent_engines.get(resource_id)
    return remote_app.create_session(user_id=user_id)


def send_agent_message(
    resource_id: str, user_id: str, session_id: str, message: str
) -> List[Dict[str, Any]]:
    """Sends a message to the agent and streams the response."""
    try:
        print(f"Sending message with params: resource_id={resource_id}, user_id={user_id}, session_id={session_id}, message={message}")
        remote_app = agent_engines.get(resource_id)
        print("Got remote app:", remote_app)
        
        # Generate a unique invocation ID
        invocation_id = str(uuid.uuid4())
        print("Generated invocation_id:", invocation_id)
        
        events = []
        for event in remote_app.stream_query(
            user_id=user_id,
            session_id=session_id,
            message=message
        ):
            # print("Event:", event)
            events.append(event)

        # print("Response events:", json.dump(events))
        return events
    except Exception as e:
        print("Error in send_agent_message:", str(e))
        raise e


# --- FastAPI Endpoints ---

@app.get("/deployments", response_model=List[DeploymentResponse], tags=["Deployments"])
def handle_list_deployments():
    """
    Lists all available agent deployments.
    """
    try:
        return list_all_deployments()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployments/create/sessions", response_model=SessionResponse, tags=["Sessions"])
def handle_create_session(session_req: SessionRequest):
    """
    Creates a new chat session for a given deployment.
    """
    try:
        print("Creating session with:", session_req.dict())
        session = create_new_session(session_req.resource_id, session_req.user_id)
        print("Raw session response:", session)
        
        # Map the response fields to match our SessionResponse model
        return {
            "session_id": str(session["id"]),
            "user_id": str(session["userId"]),
            "resource_id": str(session["appName"]),
            "last_update_time": str(session["lastUpdateTime"])
        }
    except Exception as e:
        print("Error creating session:", str(e))
        print("Error type:", type(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployments/sessions/send", response_model=MessageResponse, tags=["Sessions"])
def handle_send_message(msg_req: MessageRequest):
    """
    Sends a message to a specific session and gets the agent's response.
    """
    try:
        # print("Received message request:", msg_req.model_dump())
        response_events = send_agent_message(
            msg_req.resource_id,
            msg_req.user_id,
            msg_req.session_id,
            msg_req.msg_req
        )
        # print("Final response events:", response_events)

        if not response_events:
            # Return a valid, empty response
            return {"response": []}

        # for event in reversed(response_events):
        #     if 'content' in event and event.get('content', {}).get('role') == 'model':
        #         try:
        #             final_text = event['content']['parts'][0]['text']
        #             # --- FIX: Wrap the final text in a dictionary ---
        #             return {"response": [{"text": final_text}]}
        #         except (KeyError, IndexError, TypeError):
        #             # --- FIX: Wrap the error message in a dictionary ---
        #             return {"response": [{"error": "Error parsing response from agent"}]}
        return {"response": response_events}
        
        # --- FIX: Wrap the final message in a dictionary ---
        return {"response": [{"info": "No valid response content from agent"}]}
        
    except Exception as e:
        print("Error in handle_send_message:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
