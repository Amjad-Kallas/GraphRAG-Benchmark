import requests

class VLLMGenerator:
    def __init__(self, base_url, model, temperature=0.7, max_tokens=512):
        self.url = f"{base_url}/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(self.url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
