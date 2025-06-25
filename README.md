import httpx
import asyncio
from typing import Optional, Dict, Any

class SimpleLLMClient:
    def __init__(self, rag_service_url: str, app_id: str):
        self.rag_service_url = rag_service_url
        self.app_id = app_id
    
    async def ask_llm(
        self,
        azure_token: str,
        prompt: str,
        system_prompt: str = "You are an intelligent AI assistant",
        model_type: str = "gpt-4o-mini",
        temperature: float = 0.1,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a simple prompt to LLM without document search
        
        Args:
            azure_token: Azure authentication token
            prompt: The user prompt/question
            system_prompt: System prompt for the LLM
            model_type: Model to use (default: gpt-4o-mini)
            temperature: Temperature for response generation
            session_id: Optional session ID
            
        Returns:
            Dict containing the LLM response
        """
        try:
            headers = {
                "accept": "application/json", 
                "Authorization": azure_token
            }
            
            json_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "model": model_type,
                "temperature": temperature,
                "search_type": "NO-SEARCH",  # No document search
                "use_tools": False,
                "save_response": False,
                "multimodal": False
            }
            
            if session_id:
                json_data["session_id"] = session_id
            
            url = f"{self.rag_service_url}/{self.app_id}"
            
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, headers=headers, json=json_data)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None

# Usage example
async def main():
    # Initialize client
    client = SimpleLLMClient(
        rag_service_url="https://your-rag-service.azurewebsites.net/rag",
        app_id="your-app-id"
    )
    
    # Make a call
    response = await client.ask_llm(
        azure_token="your-azure-token",
        prompt="What is the capital of France?",
        system_prompt="You are a helpful geography assistant"
    )
    
    if response:
        print("Answer:", response.get("answer", "No answer found"))
    else:
        print("Failed to get response")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
