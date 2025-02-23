import openai
from config import MODEL_NAME

class OpenAI_LLM:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate(self, prompt: str):
        response = openai.Completion.create(
            engine=MODEL_NAME,
            prompt=prompt,
            max_tokens=50
        )
        return response.choices[0].text.strip()
