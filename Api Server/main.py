import os
import sys
from typing import List, Dict, Any, Optional
import json
import uvicorn
import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from pydantic import BaseModel, Field
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
import uuid
from google.cloud import storage
from vertexai.generative_models import Part

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
GCS_UPLOAD_BUCKET = "g2m_temporary_files" # Bucket for user file uploads

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
    resource_id: str,
    user_id: str,
    session_id: str,
    message_parts: List[Part]
) -> List[Dict[str, Any]]:
    """Sends a message (composed of multiple parts) to the agent and streams the response."""
    try:
        print(f"Sending message with params: resource_id={resource_id}, user_id={user_id}, session_id={session_id}")
        remote_app = agent_engines.get(resource_id)
        print("Got remote app:", remote_app)

        print(f"Sending message with parts: {message_parts}")
        events = []
        for event in remote_app.stream_query(
            user_id=user_id,
            session_id=session_id,
            message=message_parts
        ):
            print("Received event:", event)
            events.append(event)
        return events

    except Exception as e:
        print(f"Error in send_agent_message: {str(e)}")
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
def handle_send_message(
    user_id: str = Form(...),
    resource_id: str = Form(...),
    session_id: str = Form(...),
    msg_req: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Sends a message (and optionally a file) to a specific session and gets the agent's response.
    If a file is provided, its URI is included in a JSON payload sent to the agent.
    """
    try:
        gcs_uri = None
        # Handle file upload: upload to GCS and get the URI
        if file:
            print(f"Uploading file: {file.filename}")
            storage_client = storage.Client()
            bucket_name = "g2m_temporary_files"
            bucket = storage_client.bucket(bucket_name)

            # Create a unique name for the file in GCS to avoid collisions
            name, extension = os.path.splitext(file.filename)
            unique_filename = f"{name}-{uuid.uuid4()}{extension}"
            blob = bucket.blob(unique_filename)

            # Upload the file from the stream
            blob.upload_from_file(file.file)
            print(f"File {file.filename} uploaded to {unique_filename} in bucket {bucket_name}.")

            gcs_uri = f"gs://{bucket_name}/{unique_filename}"

        # Handle text message
        message_text = msg_req
        if file and not message_text:
            message_text = "give me a summary of the document"

        # Construct the JSON payload
        payload = {}
        if message_text:
            payload["message"] = message_text
        if gcs_uri:
            payload["gcs_uri"] = gcs_uri

        if not payload:
            raise HTTPException(status_code=400, detail="A message or a file is required.")

        # Create a single message part with the JSON payload
        message_content = json.dumps(payload)


        # Pass the parts to the core agent function
        response_events = send_agent_message(
            resource_id=resource_id,
            user_id=user_id,
            session_id=session_id,
            message_parts=message_content
        )

        if not response_events:
            return {"response": []}
        print("Response events received:", response_events)
        return {"response": response_events}

    except Exception as e:
        print(f"Error in handle_send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
