# src/agents/agent.py

class Agent:
    def __init__(self,
                 task: str,
                 prompt: str = None):
        self.task = task

        # prompt initialization
        self.prompt = prompt
        if self.prompt is None:
            self.prompt = self.initialize_prompt()
    
    def __str__(self):
        return self.task
    
    def copy(self):
        return Agent(self.task, self.prompt)
    
    def initialize_prompt(self):
        """
        Generate a prompt for the task by using the default generation prompt
        """
        raise NotImplementedError
    
    def update_prompt(self, prompt: str):
        self.prompt = prompt
    
    def run(self, input_data: str, llm_client):
        # response = llm_client.generate(system_prompt=self.prompt, user_content=input_data)
        # return response
        raise NotImplementedError
