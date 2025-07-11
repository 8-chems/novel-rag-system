import requests
import json
import os
import time
from typing import Dict, Any, Optional


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize Groq API client."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable.")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def call(self, prompt: str, model: str = "gemma2-9b-it", max_tokens: int = 1000, temperature: float = 0.7) -> \
    Dict[str, Any]:
        """Make a Groq API call with retry logic."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                # Attempt to parse content as JSON if it looks like JSON
                try:
                    return json.loads(content) if content.startswith("{") or content.startswith("[") else content
                except json.JSONDecodeError:
                    return content
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    print(f"Groq API call failed after {self.max_retries} attempts: {str(e)}")
                    return {}
                time.sleep(self.retry_delay)
        return {}


if __name__ == "__main__":
    # Example usage
    client = GroqClient(api_key="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")
    prompt = "Return a JSON object: {\"test\": \"example\"}"
    result = client.call(prompt)
    print(f"Result: {result}")