# src/llm/llm.py
from openai import OpenAI
from src.config import OPENAI_API_KEY

class LLMClient:
    def __init__(self,
                 model: str = None,
                 temperature: float = None):
        if model is None:
            raise ValueError("Model is required")
        if temperature is None:
            raise ValueError("Temperature is required")
        
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

class PromptGenerator(LLMClient):
    """
    Prompt generator for generating the initial agent and workflow prompts
    """
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.7):
        super().__init__(model, temperature)
    
    def generate_prompt(self,
                        default_prompt: str,
                        description: str) -> str:
        response = self.generate(default_prompt, description)
        return response['content']

class AgentClient(LLMClient):
    """
    Agent client for generating agent responses
    """
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.1):  # use lower temperature
        super().__init__(model, temperature)
    
    def generate_response(self,
                          prompt: str,
                          input_data: str) -> str:
        return self.generate(prompt, input_data)

if __name__ == "__main__":
    default_prompt = "You are a prompt generator. You will be given a agent description and you will generate a prompt for the agent."
    description = "The agent's role is 'Joke Generator'. You will be given a topic and you will generate a joke about it."
    input_data = "What is the capital of France?"

    print("======= [1] PromptGenerator Testing =======\n")

    print("Default prompt:")
    print(default_prompt)
    print("\nDescription:")
    print(description)
    print("\nGenerating prompt...")

    prompt_generator = PromptGenerator()
    prompt = prompt_generator.generate_prompt(default_prompt, description)

    print("\nGenerated prompt:")
    print(prompt)

    print("\n======= [2] AgentClient Testing =======\n")

    print("Input data:")
    print(input_data)

    agent_client = AgentClient()
    response = agent_client.generate_response(prompt, input_data)

    print("\nResponse:")
    print(response['content'])
    print(f"\nPrompt tokens: {response['prompt_tokens']}")
    print(f"Response tokens: {response['response_tokens']}")
    print(f"Total tokens: {response['total_tokens']}")
