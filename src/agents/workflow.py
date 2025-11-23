# src/agents/workflow.py
from agent import Agent

class Workflow(Agent):
    def __init__(self, description: str):
        self.description = description
        
        # workflow initialization
        self.workflow = []
        self.initialize_workflow()
    
    def __str__(self):
        representation = self.represent()
        return "\n".join(representation)

    def copy(self):
        return [Agent(agent.task, agent.prompt) for agent in self.workflow]
    
    def initialize_workflow(self):
        """
        Generate a workflow for the task by using the default generation prompt
        """
        self.workflow.append(Agent(task="Parse Task"))
        self.workflow.append(Agent(task="Refine Task"))
        self.workflow.append(Agent(task="Generate Code"))
        self.workflow.append(Agent(task="Review Code"))
        self.workflow.append(Agent(task="Refine Code"))
    
    def represent(self):
        """
        Represent workflow using the CoRE scheme (Xu et al., 2024b)
        """
        representation = []
        for i, agent in enumerate(self.workflow, start=1):
            representation.append(f"Step {i}:::Process:::{agent.task}:::next::Step {i + 1}")
        representation.append(f"Step {i + 1}:::Terminal:::End of Workflow:::")
        return representation
    
    def execute(self, input_data: str, llm_client):
        current_context = input_data
        for agent in self.workflow:
            current_context = agent.run(current_context, llm_client)
        return current_context
    
    def evaluate(self):
        raise NotImplementedError
