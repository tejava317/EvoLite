# src/llm/client.py
from openai import OpenAI
from src.config import API_KEY, BASE_URL


def get_client():
    """Get OpenAI client configured for OpenAI or vLLM."""
    if BASE_URL:
        return OpenAI(api_key=API_KEY, base_url=BASE_URL)
    else:
        return OpenAI(api_key=API_KEY)


def list_available_models():
    """
    List all available models from the endpoint.
    """
    client = get_client()
    models = client.models.list()
    return models.data


class LLMClient:
    def __init__(self,
                 model: str = None,
                 temperature: float = None):
        if model is None:
            raise ValueError("Model is required")
        if temperature is None:
            raise ValueError("Temperature is required")
        
        self.client = get_client()
        self.model = model
        self.temperature = temperature
    
    def generate(self,
                 system_prompt: str,
                 user_content: str,
                 max_tokens: int = 1000) -> dict:
        """Generate a response from the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        
        return {
            "content": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "response_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }
    
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
                 model: str = None,
                 temperature: float = 0.7):
        # Auto-detect model if not specified
        if model is None:
            model = self._get_default_model()
        super().__init__(model, temperature)
    
    @staticmethod
    def _get_default_model() -> str:
        """Get the first available model or fall back to gpt-4o-mini."""
        try:
            models = list_available_models()
            if models:
                return models[0].id
        except:
            pass
        return "gpt-4o-mini"
    
    def generate_prompt(self,
                        default_prompt: str,
                        description: str) -> str:
        response = self.generate(system_prompt=default_prompt, user_content=description)
        return response['content']


class AgentClient(LLMClient):
    """
    Agent client for generating agent responses
    """
    def __init__(self,
                 model: str = None,
                 temperature: float = 0.1):  # use lower temperature
        # Auto-detect model if not specified
        if model is None:
            model = self._get_default_model()
        super().__init__(model, temperature)
    
    @staticmethod
    def _get_default_model() -> str:
        """Get the first available model or fall back to gpt-4o-mini."""
        try:
            models = list_available_models()
            if models:
                return models[0].id
        except:
            pass
        return "gpt-4o-mini"
    
    def generate_response(self,
                          prompt: str,
                          input_data: str) -> str:
        return self.generate(system_prompt=prompt, user_content=input_data)


if __name__ == "__main__":
    print("======= LLM Client Testing =======\n")
    
    # List available models
    print("Fetching available models...")
    try:
        models = list_available_models()
        print(f"Available models ({len(models)}):")
        for m in models[:5]:  # Show first 5
            print(f"  - {m.id}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")
    except Exception as e:
        print(f"Could not list models: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test AgentClient
    print("Testing AgentClient...")
    try:
        agent_client = AgentClient()
        print(f"Using model: {agent_client.model}")
        
        response = agent_client.generate_response(
            prompt="You are a helpful assistant.",
            input_data="What is 2 + 2?"
        )
        
        print(f"\nResponse: {response['content']}")
        print(f"Tokens: {response['total_tokens']}")
    except Exception as e:
        print(f"Error: {e}")
