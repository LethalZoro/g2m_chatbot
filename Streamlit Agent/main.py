import streamlit as st
import requests
import uuid

# --- Configuration ---
# The base URL of your running FastAPI application
API_BASE_URL = "https://my-fastapi-service-469937863197.us-central1.run.app"

# --- API Communication Functions ---
# These functions will interact with your FastAPI endpoints.

def get_deployments():
    """Fetches the list of available agent deployments."""
    try:
        response = requests.get(f"{API_BASE_URL}/deployments")
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching deployments: {e}")
        return None

def create_session(resource_id: str, user_id: str):
    """Creates a new chat session."""
    try:
        payload = {"resource_id": resource_id, "user_id": user_id}
        response = requests.post(f"{API_BASE_URL}/deployments/create/sessions", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating a new session: {e}")
        st.error(f"Response content: {response.text}")
        return None

def send_message(resource_id: str, user_id: str, session_id: str, message: str):
    """Sends a message to the agent and gets a response."""
    try:
        payload = {
            "resource_id": resource_id,
            "user_id": user_id,
            "session_id": session_id,
            "msg_req": message,
        }
        response = requests.post(f"{API_BASE_URL}/deployments/sessions/send", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message: {e}")
        st.error(f"Response content: {response.text}")
        return None

# --- Response Parsing Function ---

def parse_agent_response(response_json: dict) -> str:
    """
    Parses the potentially complex JSON response from the agent to extract readable text.
    This function is designed to be robust and handle multiple possible response structures.
    """
    if not response_json or "response" not in response_json:
        return "[No response found]"

    response_events = response_json["response"]
    if not response_events:
        return "[Agent returned an empty response]"

    full_text = []

    # Iterate through all events in the response list
    for event in response_events:
        # First, check for the streamed 'chunk' format
        if "chunk" in event and "text" in event["chunk"]:
            full_text.append(event["chunk"]["text"])
            continue

        # Next, check for the 'content' block format as a fallback
        if "content" in event and event.get("content", {}).get("role") == "model":
            try:
                text = event['content']['parts'][0]['text']
                full_text.append(text)
            except (KeyError, IndexError, TypeError):
                # This part of the content was not in the expected format
                continue
    
    if full_text:
        return "".join(full_text)
    else:
        # If no text was extracted, show the raw JSON for debugging
        st.warning("Could not parse a text response from the agent. Displaying raw JSON:")
        st.json(response_json)
        return "[Error parsing agent response]"


# --- Main Streamlit App Logic ---

st.set_page_config(page_title="Vertex Agent Chat", layout="wide")

st.title("Vertex AI Agent Chatbot")
st.write("A Streamlit interface to test and interact with a Vertex AI Agent Engine.")

# --- Initialization & Session State ---
# Use session_state to persist data across reruns

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "resource_id" not in st.session_state:
    st.session_state.resource_id = None
if "user_id" not in st.session_state:
    # Generate a unique user ID for this browser session
    st.session_state.user_id = f"st-user-{str(uuid.uuid4())}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False


def initialize_agent():
    """
    Fetches deployments, selects the first one, and creates a new session.
    This is the main setup function.
    """
    with st.spinner("Initializing Agent..."):
        deployments = get_deployments()
        if deployments:
            # Extract the agent ID from the full resource name
            # e.g., "projects/.../agentEngines/12345" -> "12345"
            full_resource_name = deployments[0]["resource_name"]
            st.session_state.resource_id = full_resource_name.split('/')[-1]
            
            # Create the first session
            create_and_start_new_session()
            st.session_state.agent_initialized = True
        else:
            st.error("Could not initialize agent. Please ensure the FastAPI backend is running and configured correctly.")
            st.session_state.agent_initialized = False

def create_and_start_new_session():
    """Clears chat and creates a new session."""
    if not st.session_state.resource_id:
        st.warning("Cannot create a new session without a Resource ID.")
        return
        
    session_info = create_session(st.session_state.resource_id, st.session_state.user_id)
    if session_info:
        st.session_state.session_id = session_info["session_id"]
        # Reset chat history
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to help. How can I assist you today?"}]
        st.success(f"New session created: {st.session_state.session_id}")
    else:
        st.error("Failed to create a new session.")


# Run initialization only once on first load
if not st.session_state.agent_initialized:
    initialize_agent()


# --- UI Components ---

# Sidebar for controls and debug info
with st.sidebar:
    st.header("Controls & Info")
    if st.button("Create New Session", type="primary"):
        create_and_start_new_session()

    st.markdown("---")
    st.subheader("Debug Information")
    st.write(f"**User ID:** `{st.session_state.user_id}`")
    st.write(f"**Agent Resource ID:** `{st.session_state.resource_id}`")
    st.write(f"**Current Session ID:** `{st.session_state.session_id}`")


# Main chat interface
if st.session_state.agent_initialized:
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for user
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_json = send_message(
                    st.session_state.resource_id,
                    st.session_state.user_id,
                    st.session_state.session_id,
                    prompt
                )
                
                if response_json:
                    # Parse the response to get clean text
                    assistant_response = parse_agent_response(response_json)
                    st.markdown(assistant_response)
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    error_message = "Failed to get a response from the agent."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.warning("Agent is not initialized. Please check the backend connection and refresh the page.")

