# src/llm/llm.py
from openai import OpenAI
from src.config import OPENAI_API_KEY

class LLMClient:
    """
    LLM client for generating the initial agent and workflow (using OpenAI)
    """
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.7):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature
    
    def generate(self,
                 system_prompt: str,
                 user_content: str,
                 max_tokens: int = 1000) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            return {
                "content": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "response_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        except Exception as e:
            raise Exception(f"Error: generating response: {str(e)}")
    
    def generate_batch(self,
                      system_prompt: str,
                      user_contents: list[str],
                      max_tokens: int = 1000) -> list[dict]:
        responses = []
        for user_content in user_contents:
            response = self.generate(system_prompt, user_content, max_tokens)
            responses.append(response)
        return responses

if __name__ == "__main__":
    system_prompt = "You are a joke generator. You will be given a topic and you will generate a joke about it."
    user_content = "What is the capital of France?"
    
    client = LLMClient()
    response = client.generate(system_prompt, user_content)
    print(response)
