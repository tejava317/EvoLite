# src/llm/client.py
"""
LLM Client supporting both OpenAI and RunPod custom servers.
"""
from openai import OpenAI
from src.config import API_KEY, BASE_URL, USE_RUNPOD


def get_client():
    """
    Get OpenAI client configured for either RunPod or OpenAI.
    """
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
        self._use_runpod = USE_RUNPOD
    
    def generate(self,
                 system_prompt: str,
                 user_content: str,
                 max_tokens: int = 1000) -> dict:
        """
        Generate a response from the LLM.
        
        Handles differences between OpenAI and RunPod APIs.
        """
        try:
            # Build request parameters
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": self.temperature,
            }
            
            # Some models (like o1/o3) use max_completion_tokens instead of max_tokens
            # Try max_completion_tokens first for RunPod, fall back to max_tokens
            if self._use_runpod:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**params)
            
            return {
                "content": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "response_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        except Exception as e:
            error_str = str(e)
            # Handle the case where max_completion_tokens isn't supported
            if "max_completion_tokens" in error_str and "max_tokens" not in params:
                # Retry with max_tokens
                params.pop("max_completion_tokens", None)
                params["max_tokens"] = max_tokens
                try:
                    response = self.client.chat.completions.create(**params)
                    return {
                        "content": response.choices[0].message.content,
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "response_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                except Exception as e2:
                    raise Exception(f"Error generating response: {str(e2)}")
            raise Exception(f"Error generating response: {str(e)}")
    
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
        response = self.generate(default_prompt, description)
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
        return self.generate(prompt, input_data)


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
