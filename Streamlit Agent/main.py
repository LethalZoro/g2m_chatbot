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
        if 'response' in locals() and response.text:
            st.error(f"Response content: {response.text}")
        return None

def send_message(resource_id: str, user_id: str, session_id: str, message: str, uploaded_file=None):
    """Sends a message and optionally a file to the agent and gets a response."""
    try:
        # The payload is now sent as form data, not JSON
        data = {
            "resource_id": resource_id,
            "user_id": user_id,
            "session_id": session_id,
            "msg_req": message,
        }
        
        files = None
        if uploaded_file:
            # If a file is attached, prepare it for the multipart request
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

        # The 'requests' library automatically sets the correct Content-Type
        # to 'multipart/form-data' when the 'files' argument is used.
        response = requests.post(
            f"{API_BASE_URL}/deployments/sessions/send", 
            data=data, 
            files=files
        )
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message: {e}")
        if 'response' in locals() and response.text:
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
if "deployments" not in st.session_state:
    st.session_state.deployments = None
if "agent_selected" not in st.session_state:
    st.session_state.agent_selected = False


def start_new_chat_session(resource_id):
    """Clears chat history and creates a new session for the selected agent."""
    with st.spinner(f"Creating a new session for agent..."):
        session_info = create_session(resource_id, st.session_state.user_id)
        if session_info and "session_id" in session_info:
            st.session_state.session_id = session_info["session_id"]
            st.session_state.resource_id = resource_id
            st.session_state.agent_selected = True
            # Reset chat history for the new session
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to help. How can I assist you today?"}]
            st.success(f"New session created: {st.session_state.session_id}")
            # Rerun to update the main UI
            st.rerun()
        else:
            st.error("Failed to create a new session. Please check the backend logs.")
            st.session_state.agent_selected = False


# --- Initial Fetching of Deployments ---
if st.session_state.deployments is None:
    with st.spinner("Fetching available agent deployments..."):
        st.session_state.deployments = get_deployments()
        if st.session_state.deployments is None:
            st.error("Could not fetch deployments. Please ensure the FastAPI backend is running and configured correctly.")
            # Stop the script if we can't get deployments
            st.stop()

# --- UI Components ---

# Sidebar for controls and debug info
with st.sidebar:
    st.header("Controls & Info")
    # This button will only be active after a session has started
    if st.session_state.agent_selected:
        if st.button("Start New Chat", type="primary"):
            # Create a new session with the *same* agent
            start_new_chat_session(st.session_state.resource_id)

    st.markdown("---")
    st.subheader("Debug Information")
    st.write(f"**User ID:** `{st.session_state.user_id}`")
    st.write(f"**Agent Resource ID:** `{st.session_state.resource_id}`")
    st.write(f"**Current Session ID:** `{st.session_state.session_id}`")


# --- Main Interface Logic ---

# If an agent hasn't been selected yet, show the selection dropdown.
if not st.session_state.agent_selected:
    st.header("1. Select an Agent Deployment")

    # STEP 1: DEFINE YOUR CUSTOM NAMES HERE
    # This is the only place you need to make changes.
    # The "key" is the long, technical resource_name from Google.
    # The "value" is the pretty, user-friendly name you want to show.
    AGENT_DISPLAY_NAMES = {
        # Technical Name (from your API)         : User-Facing Name (what you want to show)
        
        "projects/469937863197/locations/us-central1/reasoningEngines/1687979047054737408": "Ask-G2M Agent With BigQuery and Google Search",
        "projects/469937863197/locations/us-central1/reasoningEngines/1084496696987090944": "Ask-G2M Agent With BigQuery Gemini 2.5 Pro",
        "projects/469937863197/locations/us-central1/reasoningEngines/5418085438424350720": "Ask-G2M Agent Gemini 2.5 Pro",
        "projects/469937863197/locations/us-central1/reasoningEngines/7737579984008511488": "Ask-G2M Agent Gemini 2.5 Flash"
    }

    # STEP 2: THE CODE BUILDS THE OPTIONS FOR THE DROPDOWN
    # It will create a new dictionary for the UI, flipping the key and value.
    # For example: {"Sales Support Agent": "projects/..."}
    deployment_options = {}
    if st.session_state.deployments:
        for dep in st.session_state.deployments:
            resource_name = dep['resource_name']
            # fallback_name = f"Agent (Unmapped) {resource_name.split('/')[-1]}"
            
            # Look up the pretty name from your dictionary above
            display_name = AGENT_DISPLAY_NAMES.get(resource_name)
            
            # The options for the dropdown will be based on the pretty name
            deployment_options[display_name] = resource_name

    if not deployment_options:
        st.warning("No agent deployments found.")
        st.stop()

    # STEP 3: SHOW THE CUSTOM NAMES TO THE USER
    # The 'options' parameter is given `list(deployment_options.keys())`,
    # which are the pretty names you defined (e.g., "Sales Support Agent", "Technical FAQ Bot").
    # The user NEVER sees the long technical name in this list.
    selected_display_name = st.selectbox(
        "Choose an available agent to chat with:",
        options=list(deployment_options.keys()), # This uses YOUR custom names!
        index=None,
        placeholder="Select agent..."
    )

    # STEP 4: WHEN THE USER SELECTS A NAME, THE CODE FINDS THE CORRECT TECHNICAL ID
    if selected_display_name:
        # The user sees and selects "Sales Support Agent", but the code gets the
        # corresponding technical name (`projects/...`) to use with the API.
        selected_resource_id = deployment_options[selected_display_name]
        
        st.header("2. Start Chatting")
        st.info(f"You have selected: **{selected_display_name}**")
        
        if st.button(f"Begin Chat with {selected_display_name}", type="primary"):
            start_new_chat_session(selected_resource_id)


# If an agent IS selected and a session is active, show the chat interface.
if st.session_state.agent_selected:
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ID of the agent that supports file uploads
    FILE_UPLOAD_AGENT_ID = "projects/469937863197/locations/us-central1/reasoningEngines/1029046126575091712"

    # --- Conditionally show the correct input based on the selected agent ---
    if st.session_state.resource_id == FILE_UPLOAD_AGENT_ID:
        # --- Chat Input Form for File Upload Agent ---
        with st.form(key="chat_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Attach a file (optional)", type=['pdf', 'docx', 'txt'])
            prompt = st.text_input("What would you like to ask?", key="chat_input_box")
            submit_button = st.form_submit_button(label='Send')

        if submit_button and (prompt or uploaded_file):
            user_message_content = prompt
            if uploaded_file:
                user_message_content += f"\n*(Attached file: `{uploaded_file.name}`)*"
            
            st.session_state.messages.append({"role": "user", "content": user_message_content})
            with st.chat_message("user"):
                st.markdown(user_message_content)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_json = send_message(
                        st.session_state.resource_id,
                        st.session_state.user_id,
                        st.session_state.session_id,
                        prompt,
                        uploaded_file=uploaded_file
                    )
                    
                    if response_json:
                        assistant_response = parse_agent_response(response_json)
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    else:
                        error_message = "Failed to get a response from the agent."
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.rerun()

    else:
        # --- Standard Chat Input for other agents ---
        if prompt := st.chat_input("What would you like to ask?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_json = send_message(
                        st.session_state.resource_id,
                        st.session_state.user_id,
                        st.session_state.session_id,
                        prompt
                    )
                    
                    if response_json:
                        assistant_response = parse_agent_response(response_json)
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    else:
                        error_message = "Failed to get a response from the agent."
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})