import os
import sys
from typing import List, Dict, Any, Optional
import json
import uvicorn
import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

# --- CORS Configuration ---
# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

class SessionListRequest(BaseModel):
    user_id: str = Field(..., example="abc1")
    resource_id: str = Field(..., example="3976792820177436672")

class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]] = Field(..., example=[{"session_id": "123", "last_update_time": "1750166696.884852"}])

class ChatHistoryRequest(BaseModel):
    resource_id: str = Field(..., example="3976792820177436672")
    user_id: str = Field(..., example="abc1")
    session_id: str = Field(..., example="3217044029279567872")

class ChatHistoryResponse(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., example=[{"role": "user", "content": "Hello"}, {"role": "agent", "content": "Hi there!"}])


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


def list_user_sessions(resource_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Lists all sessions for a specific user in a deployment."""
    try:
        print(f"Listing sessions for user_id={user_id}, resource_id={resource_id}")
        remote_app = agent_engines.get(resource_id)
        sessions_response = remote_app.list_sessions(user_id=user_id)
        
        print(f"Raw sessions response: {sessions_response}")
        print(f"Sessions response type: {type(sessions_response)}")
        
        # Extract the sessions list from the response
        if isinstance(sessions_response, dict) and 'sessions' in sessions_response:
            sessions = sessions_response['sessions']
        elif isinstance(sessions_response, list):
            # If it's already a list, use it directly
            sessions = sessions_response
        else:
            print(f"Unexpected sessions response format: {type(sessions_response)}")
            return []
        
        # Handle different response formats for the sessions list
        if not isinstance(sessions, list):
            if sessions is not None:  # Make sure it's not None
                sessions = [sessions]
            else:
                print("No sessions found (None response)")
                return []
        
        # Format the sessions data for the response
        formatted_sessions = []
        for session in sessions:
            if hasattr(session, '__dict__') or isinstance(session, dict):
                # Handle both dict and object types
                if isinstance(session, dict):
                    session_data = session
                else:
                    # Convert object to dict if it has attributes
                    session_data = {
                        'id': getattr(session, 'id', ''),
                        'userId': getattr(session, 'userId', ''),
                        'appName': getattr(session, 'appName', ''),
                        'lastUpdateTime': getattr(session, 'lastUpdateTime', '')
                    }
                
                formatted_sessions.append({
                    "session_id": str(session_data.get("id", "")),
                    "user_id": str(session_data.get("userId", "")),
                    "resource_id": str(session_data.get("appName", "")),
                    "last_update_time": str(session_data.get("lastUpdateTime", ""))
                })
            else:
                print(f"Unexpected session item format: {type(session)}, value: {session}")
        
        return formatted_sessions
    except Exception as e:
        print(f"Error in list_user_sessions: {str(e)}")
        print(f"Error type: {type(e)}")
        raise e


def get_session_chat_history(resource_id: str, user_id: str, session_id: str) -> List[Dict[str, Any]]:
    """Gets the chat history for a specific session."""
    try:
        print(f"Getting chat history for session_id={session_id}, user_id={user_id}, resource_id={resource_id}")
        remote_app = agent_engines.get(resource_id)
        
        # Try different methods to get session data
        session_data = None
        chat_history = []
        
        # Method 1: Try to get session directly and inspect its structure
        try:
            if hasattr(remote_app, 'get_session'):
                session = remote_app.get_session(user_id=user_id, session_id=session_id)
                print(f"Session object: {session}")
                print(f"Session type: {type(session)}")
                
                if session:
                    print(f"Session has keys: {list(session.keys()) if isinstance(session, dict) else 'Not a dict'}")
                    
                    # Method 1A: Check if session.state.chat_history exists directly
                    if isinstance(session, dict) and 'state' in session and 'chat_history' in session['state']:
                        print("Found chat_history in session.state!")
                        extracted_history = session['state']['chat_history']
                        if isinstance(extracted_history, list):
                            for msg in extracted_history:
                                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                    chat_history.append({
                                        "role": msg['role'],
                                        "content": msg['content'],
                                        "timestamp": session.get('lastUpdateTime', ''),
                                        "message_id": session.get('id', '')
                                    })
                            if chat_history:
                                print(f"Extracted {len(chat_history)} messages from session.state.chat_history")
                                return chat_history
                    
                    # Method 1B: Look through events for the most recent chat_history
                    if isinstance(session, dict) and 'events' in session:
                        events = session['events']
                        print(f"Found {len(events)} events in session")
                        
                        # Go through events in reverse order to get the most recent chat_history
                        for event in reversed(events):
                            if isinstance(event, dict) and 'actions' in event:
                                actions = event['actions']
                                if 'stateDelta' in actions and 'chat_history' in actions['stateDelta']:
                                    print(f"Found chat_history in event from {event.get('author', 'unknown')}")
                                    extracted_history = actions['stateDelta']['chat_history']
                                    if isinstance(extracted_history, list):
                                        for msg in extracted_history:
                                            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                                chat_history.append({
                                                    "role": msg['role'],
                                                    "content": msg['content'],
                                                    "timestamp": event.get('timestamp', ''),
                                                    "message_id": event.get('id', '')
                                                })
                                        if chat_history:
                                            print(f"Extracted {len(chat_history)} messages from event chat_history")
                                            return chat_history
                                elif 'state_delta' in actions and 'chat_history' in actions['state_delta']:
                                    print(f"Found chat_history in event state_delta from {event.get('author', 'unknown')}")
                                    extracted_history = actions['state_delta']['chat_history']
                                    if isinstance(extracted_history, list):
                                        for msg in extracted_history:
                                            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                                chat_history.append({
                                                    "role": msg['role'],
                                                    "content": msg['content'],
                                                    "timestamp": event.get('timestamp', ''),
                                                    "message_id": event.get('id', '')
                                                })
                                        if chat_history:
                                            print(f"Extracted {len(chat_history)} messages from event state_delta")
                                            return chat_history
                        
                        # Method 1C: If no chat_history found in events, extract from content fields
                        print("No chat_history found in events, trying to extract from content...")
                        for event in events:
                            if isinstance(event, dict) and 'content' in event and event['content']:
                                content = event['content']
                                role = content.get('role', 'unknown')
                                
                                if 'parts' in content and isinstance(content['parts'], list):
                                    # Extract text from parts
                                    text_parts = []
                                    for part in content['parts']:
                                        if isinstance(part, dict) and 'text' in part:
                                            text_parts.append(part['text'])
                                    
                                    if text_parts:
                                        combined_text = ' '.join(text_parts)
                                        chat_history.append({
                                            "role": role,
                                            "content": combined_text,
                                            "timestamp": event.get('timestamp', ''),
                                            "message_id": event.get('id', '')
                                        })
                        
                        if chat_history:
                            print(f"Extracted {len(chat_history)} messages from event content")
                            return chat_history
                        
        except Exception as session_err:
            print(f"Session method failed: {session_err}")
            import traceback
            traceback.print_exc()
        
        # Method 2: Try list_messages if available
        try:
            if hasattr(remote_app, 'list_messages'):
                print("Trying list_messages method...")
                messages = remote_app.list_messages(user_id=user_id, session_id=session_id)
                print(f"Raw messages response: {messages}")
                print(f"Messages type: {type(messages)}")
                
                if messages is not None:
                    # Handle different response formats
                    messages_list = messages if isinstance(messages, list) else [messages]
                    
                    for message in messages_list:
                        print(f"Processing message: {message} (type: {type(message)})")
                        
                        if isinstance(message, dict):
                            # Look for chat_history in the response structure
                            if 'actions' in message and 'state_delta' in message['actions']:
                                state_delta = message['actions']['state_delta']
                                if 'chat_history' in state_delta:
                                    extracted_history = state_delta['chat_history']
                                    if isinstance(extracted_history, list):
                                        for msg in extracted_history:
                                            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                                chat_history.append({
                                                    "role": msg['role'],
                                                    "content": msg['content'],
                                                    "timestamp": message.get('timestamp', ''),
                                                    "message_id": message.get('id', '')
                                                })
                            # Also check for content field directly
                            elif 'content' in message and 'role' in message:
                                chat_history.append({
                                    "role": message['role'],
                                    "content": str(message['content']),
                                    "timestamp": message.get('timestamp', ''),
                                    "message_id": message.get('id', '')
                                })
                        elif hasattr(message, '__dict__'):
                            # Handle object format
                            print(f"Message object attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")
                            if hasattr(message, 'content') and hasattr(message, 'role'):
                                chat_history.append({
                                    "role": getattr(message, 'role'),
                                    "content": str(getattr(message, 'content')),
                                    "timestamp": getattr(message, 'timestamp', ''),
                                    "message_id": getattr(message, 'id', '')
                                })
                    
                    if chat_history:
                        return chat_history
            else:
                print("list_messages method not available")
                
        except Exception as list_err:
            print(f"List messages method failed: {list_err}")
            import traceback
            traceback.print_exc()
        
        # Method 3: Try to get the latest interaction to see the response format
        try:
            # Check what methods are available on the remote_app
            available_methods = [method for method in dir(remote_app) if not method.startswith('_')]
            print(f"Available methods on AgentEngine: {available_methods}")
            
            # Try some common method names for getting session state/history
            for method_name in ['get_conversation_history', 'conversation_history', 'get_chat_history', 'chat_history', 'get_session_state', 'session_state']:
                if hasattr(remote_app, method_name):
                    print(f"Trying method: {method_name}")
                    try:
                        method = getattr(remote_app, method_name)
                        result = method(user_id=user_id, session_id=session_id)
                        print(f"Result from {method_name}: {result}")
                        # Parse the result based on the structure you showed
                        if isinstance(result, list):
                            for item in result:
                                if isinstance(item, dict) and 'actions' in item:
                                    actions = item['actions']
                                    if 'state_delta' in actions and 'chat_history' in actions['state_delta']:
                                        extracted_history = actions['state_delta']['chat_history']
                                        if isinstance(extracted_history, list):
                                            for msg in extracted_history:
                                                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                                    chat_history.append({
                                                        "role": msg['role'],
                                                        "content": msg['content'],
                                                        "timestamp": item.get('timestamp', ''),
                                                        "message_id": item.get('id', '')
                                                    })
                            if chat_history:
                                return chat_history
                    except Exception as method_err:
                        print(f"Method {method_name} failed: {method_err}")
                        continue
        except Exception as method_discovery_err:
            print(f"Method discovery failed: {method_discovery_err}")
        
        # If no chat history found, return empty list with helpful debug info
        print("No chat history found. This might be because:")
        print("1. The session has no messages yet")
        print("2. The AgentEngine doesn't store chat history")
        print("3. Chat history is only available in the response stream")
        return []
        
    except Exception as e:
        print(f"Error in get_session_chat_history: {str(e)}")
        print(f"Error type: {type(e)}")
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


@app.post("/sessions", response_model=SessionListResponse, tags=["Sessions"])
def handle_list_user_sessions(session_req: SessionListRequest):
    """
    Lists all sessions for a specific user in a deployment.
    
    Args:
        session_req: Request containing user_id and resource_id
    
    Returns:
        List of sessions for the user
    """
    try:
        sessions = list_user_sessions(resource_id=session_req.resource_id, user_id=session_req.user_id)
        return {"sessions": sessions}
    except Exception as e:
        print(f"Error listing user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/messages", response_model=ChatHistoryResponse, tags=["Sessions"])
def handle_get_chat_history(chat_req: ChatHistoryRequest):
    """
    Gets the chat history for a specific session.
    
    Args:
        chat_req: Request containing session_id, user_id, and resource_id
    
    Returns:
        List of messages in the session
    """
    try:
        messages = get_session_chat_history(
            resource_id=chat_req.resource_id,
            user_id=chat_req.user_id,
            session_id=chat_req.session_id
        )
        return {"messages": messages}
    except Exception as e:
        print(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
