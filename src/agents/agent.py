# src/agents/agent.py
from src.config import DEFAULT_PROMPTS
from src.llm.client import PromptGenerator, AgentClient

class Agent:
    def __init__(self,
                 role: str,
                 prompt: str = None,
                 workflow_description: str = None,
                 agent_client: AgentClient = None):
        self.role = role

        # agent client initialization
        self.agent_client = agent_client if agent_client is not None else AgentClient()

        # prompt initialization
        self.workflow_description = workflow_description
        self.prompt = prompt if prompt is not None else self.initialize_prompt()
        
        # evaluation
        self.response = None
        self.prompt_tokens = None
        self.response_tokens = None
        self.total_tokens = None
    
    def __str__(self):
        return self.role
    
    def copy(self):
        return Agent(self.role, self.prompt, self.workflow_description, self.agent_client)
    
    def initialize_prompt(self):
        """
        Generate a prompt for the agent by using the default generation prompt
        """
        default_prompt = DEFAULT_PROMPTS.get('DefaultAgentGenerationPrompt', {}).get('prompt', '')
        
        if not default_prompt:
            raise ValueError("DefaultAgentGenerationPrompt not found in configuration")
        
        if self.workflow_description:
            default_prompt = default_prompt.replace('[Workflow description]', self.workflow_description)
        
        agent_description = f"The agent's role is {self.role}. You will generate a prompt for the agent based on the role."

        prompt_generator = PromptGenerator()
        prompt = prompt_generator.generate_prompt(
            default_prompt=default_prompt,
            description=agent_description
        )
        return prompt
    
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
        response = self.agent_client.generate_response(prompt=self.prompt, input_data=input_data)

        # update evaluation
        self.response = response['content']
        self.prompt_tokens = response['prompt_tokens']
        self.response_tokens = response['response_tokens']
        self.total_tokens = response['total_tokens']
        return response

if __name__ == "__main__":
    role = "Code Generation Agent"
    workflow_description = "Task Parsing Agent -> Code Generation Agent -> Code Reviewer Agent"
    input_data = "Write a Python function that calculates the factorial of a number."

    print("======= Agent Testing =======\n")

    print("Initializing agent...")
    print(f"\nRole: {role}")
    print(f"Workflow description: {workflow_description}")

    agent = Agent(role=role, workflow_description=workflow_description)

    print("\nPrompt:")
    print(agent.prompt)
    print("\nInput data:")
    print(input_data)
    print("\nRunning agent...")

    response = agent.run(input_data)

    print("\nResponse:")
    print(response['content'])
    print(f"\nPrompt tokens: {response['prompt_tokens']}")
    print(f"Response tokens: {response['response_tokens']}")
    print(f"Total tokens: {response['total_tokens']}")
