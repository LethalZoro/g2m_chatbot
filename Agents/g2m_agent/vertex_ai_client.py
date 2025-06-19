import os
import json
import requests
from typing import Dict, Any, Optional
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VertexAIReasoningEngineClient:
    """
    Client for interacting with Vertex AI Reasoning Engine
    """
    
    def __init__(self, project_id: str = "g2m-dev", location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.base_url = f"https://{location}-aiplatform.googleapis.com/v1"
        self.reasoning_engine_id = "603878174253645824"
        
        # Initialize authentication
        self.credentials = self._get_credentials()
        self.headers = self._get_headers()
    
    def _get_credentials(self):
        """Get Google Cloud credentials"""
        try:
            # Try to use service account key if provided
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if service_account_path and os.path.exists(service_account_path):
                return service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            else:
                # Use default credentials (Application Default Credentials)
                credentials, _ = default()
                return credentials
        except Exception as e:
            print(f"Error getting credentials: {e}")
            return None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        if not self.credentials:
            raise ValueError("No valid credentials found")
        
        # Refresh credentials if needed
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        
        return {
            "Authorization": f"Bearer {self.credentials.token}",
            "Content-Type": "application/json"
        }
    
    def query_reasoning_engine(self, 
                              query: str, 
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the Vertex AI Reasoning Engine
        
        Args:
            query: The user's query/question
            context: Optional context data to provide to the reasoning engine
            
        Returns:
            Response from the reasoning engine
        """
        url = f"{self.base_url}/projects/{self.project_id}/locations/{self.location}/reasoningEngines/{self.reasoning_engine_id}:query"
        
        # Prepare the request payload
        payload = {
            "query": query
        }
        
        # Add context if provided
        if context:
            payload["context"] = context
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to reasoning engine: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return {"error": str(e)}
    
    def get_reasoning_engine_info(self) -> Dict[str, Any]:
        """Get information about the reasoning engine"""
        url = f"{self.base_url}/projects/{self.project_id}/locations/{self.location}/reasoningEngines/{self.reasoning_engine_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting reasoning engine info: {e}")
            return {"error": str(e)}


def main():
    """Example usage of the Vertex AI Reasoning Engine client"""
    
    # Initialize the client
    client = VertexAIReasoningEngineClient()
    
    # Example 1: Get reasoning engine information
    print("=== Getting Reasoning Engine Info ===")
    engine_info = client.get_reasoning_engine_info()
    print(json.dumps(engine_info, indent=2))
    
    # Example 2: Query the reasoning engine
    print("\n=== Querying Reasoning Engine ===")
    test_query = "What is the main purpose of this system?"
    
    # You can also provide context if needed
    context = {
        "user_id": "example_user",
        "session_id": "example_session"
    }
    
    response = client.query_reasoning_engine(test_query, context=context)
    print(f"Query: {test_query}")
    print(f"Response: {json.dumps(response, indent=2)}")


if __name__ == "__main__":
    main() 