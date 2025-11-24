# src/agents/agent.py
from src.config import DEFAULT_PROMPTS
from src.llm.client import LLMClient

class Agent:
    def __init__(self,
                 task: str,
                 prompt: str = None,
                 llm_client: LLMClient = None):
        self.task = task

        # LLM client initialization
        self.llm_client = llm_client if llm_client is not None else LLMClient()

        # prompt initialization
        self.prompt = prompt
        if self.prompt is None:
            self.prompt = self.initialize_prompt()
        
        # evaluation
        self.response = None
        self.prompt_tokens = None
        self.response_tokens = None
        self.total_tokens = None
    
    def __str__(self):
        return self.task
    
    def copy(self):
        return Agent(self.task, self.prompt, self.llm_client)
    
    def initialize_prompt(self):
        """
        Generate a prompt for the task by using the default generation prompt
        """
        default_prompt = DEFAULT_PROMPTS.get('DefaultAgentGenerationPrompt', {}).get('prompt', '')
        
        if not default_prompt:
            raise ValueError("DefaultAgentGenerationPrompt not found in configuration")
        
        return default_prompt
    
    def update_prompt(self, prompt: str):
        self.prompt = prompt

        self.response = None
        self.prompt_tokens = None
        self.response_tokens = None
        self.total_tokens = None
    
    def run(self, input_data: str):
        """
        Run the agent with the given input data
        """
        response = self.llm_client.generate(system_prompt=self.prompt, user_content=input_data)

        # update evaluation
        self.response = response['content']
        self.prompt_tokens = response['prompt_tokens']
        self.response_tokens = response['response_tokens']
        self.total_tokens = response['total_tokens']
        return response

if __name__ == "__main__":
    agent = Agent(task='Code Generation Agent')
    print(agent.prompt)

    input_data = "Write a Python function that calculates the factorial of a number."
    response = agent.run(input_data)
    print(response)
