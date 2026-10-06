from abc import ABC, abstractmethod
import requests
import json

class AIProvider(ABC):
    @abstractmethod
    def generate_structured_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> str:
        """
        Sends prompts to the LLM, enforcing the provided JSON schema.
        Returns the raw string output from the model.
        """
        pass

class OllamaProvider(AIProvider):
    def __init__(self, model_name: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate_structured_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": json_schema,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama provider failed: {e}")
