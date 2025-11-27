# src/agents/workflow.py
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END
from src.config import TASK_DESCRIPTIONS, DEFAULT_PROMPTS
from src.agents.agent import Agent
from src.llm.client import PromptGenerator

class WorkflowState(TypedDict):
    """State that gets passed between agents in the workflow"""
    input: str
    current_output: str
    intermediate_results: Annotated[List[str], operator.add]
    prompt_tokens: Annotated[int, operator.add]
    response_tokens: Annotated[int, operator.add]
    total_tokens: Annotated[int, operator.add]

class Workflow:
    def __init__(self,
                 task_name: str,
                 agents: List[Agent] = None):
        self.task_name = task_name
        
        # workflow initialization
        self.agents = agents if agents is not None else []
        self.memory = {}

        self.workflow_graph = None
        self.compiled_graph = None

        # build the graph if workflow is provided
        if self.agents:
            self._build_graph()

        # evaluation
        self.response = None
        self.prompt_tokens = None
        self.response_tokens = None
        self.total_tokens = None
    
    def initialize_prompt(self):
        """
        Generate a prompt for the workflow by using the default generation prompt
        """
        default_prompt = DEFAULT_PROMPTS.get('DefaultWorkflowGenerationPrompt', {}).get('prompt', '')

        if not default_prompt:
            raise ValueError("DefaultWorkflowGenerationPrompt not found in configuration")
        
        task_description = TASK_DESCRIPTIONS.get(self.task_name, {}).get('description', '')

        if not task_description:
            task_description = "This workflow is for {self.task_name} task."
        
        default_prompt = default_prompt.replace('[Dataset description]', task_description)
        return default_prompt
    
    def add_agent(self, agent: Agent):
        """
        Add a new agent to workflow
        """
        self.agents.append(agent)
        self._build_graph()
    
    # Insert an agent at the specific poistion.
    # Aware that the new agent's role becomes an argument for an ease of implementation.
    def insert_agent(self, role: str, position: int):
        
        if position < 0 or position > len(self.agents):
            raise IndexError("Position out of range while inserting.")
        
        # Generate an agent and insert at the position.
        workflow_description = self.agents[0].workflow_description if len(self.agents) else ""
        self.agents.insert(position, Agent(role=role, workflow_description=workflow_description))
        
        # Build a graph.
        self._build_graph()

        # Update the task description.
        new_task_description = " -> ".join(agent.role for agent in self.agents)
        for agent in self.agents:
            agent.task_description = new_task_description

    # Remove an agent at the specific position.
    def remove_agent(self, position: int):

        # Remove an agent.
        if position < 0 or position >= len(self.agents):
            raise IndexError("Position out of range while removing.")
        removed = self.agents.pop(position)
        
        # Build a graph.
        self._build_graph()

        # Update the task description.
        new_task_description = " -> ".join(agent.role for agent in self.agents)
        for agent in self.agents:
            agent.task_description = new_task_description


    def _create_agent_node(self, agent: Agent, agent_idx: int):
        def agent_node(state: WorkflowState) -> WorkflowState:
            input_data = state["current_output"] if state.get("current_output") else state["input"]
            response = agent.run(input_data)
            return {
                "current_output": response['content'],
                "intermediate_results": [f"Agent {agent_idx} ({agent.role}): {response['content'][:100]}..."],
                "prompt_tokens": response.get('prompt_tokens', 0),
                "response_tokens": response.get('response_tokens', 0),
                "total_tokens": response.get('total_tokens', 0)
            }
        return agent_node
    
    def _build_graph(self):
        workflow = StateGraph(WorkflowState)
        for idx, agent in enumerate(self.agents):
            node_name = f"agent_{idx}"
            workflow.add_node(node_name, self._create_agent_node(agent, idx))
        
        if self.agents:
            workflow.set_entry_point("agent_0")
            for idx in range(len(self.agents) - 1):
                workflow.add_edge(f"agent_{idx}", f"agent_{idx + 1}")
            workflow.add_edge(f"agent_{len(self.agents) - 1}", END)
        
        self.workflow_graph = workflow
        self.compiled_graph = workflow.compile()
    
    def run(self, input_data: str) -> dict:
        if not self.compiled_graph:
            raise ValueError("Workflow has no agents. Add agents before running the workflow.")
        
        initial_state = {
            "input": input_data,
            "current_output": "",
            "intermediate_results": [],
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }
        
        final_state = self.compiled_graph.invoke(initial_state)
        
        self.response = final_state["current_output"]
        self.prompt_tokens = final_state["prompt_tokens"]
        self.response_tokens = final_state["response_tokens"]
        self.total_tokens = final_state["total_tokens"]
        
        return {
            "content": final_state["current_output"],
            "intermediate_results": final_state["intermediate_results"],
            "prompt_tokens": final_state["prompt_tokens"],
            "response_tokens": final_state["response_tokens"],
            "total_tokens": final_state["total_tokens"]
        }
    
    def __str__(self):
        """String representation of the workflow"""
        agent_names = [f"{i+1}. {agent.role}" for i, agent in enumerate(self.agents)]
        return f"Workflow for {self.task_name}:\n" + "\n".join(agent_names)
    
    def copy(self):
        """Create a copy of the workflow"""
        copied_agents = [agent.copy() for agent in self.agents]
        return Workflow(self.task_name, copied_agents)

if __name__ == "__main__":
    task_name = "HumanEval"
    role1 = "Task Parsing Agent"
    role2 = "Task Refinement Agent"
    role3 = "Code Generation Agent"
    role4 = "Code Reviewer Agent"
    role5 = "Code Refinement Agent"
    input_data = "Write a Python function that calculates the factorial of a number."

    workflow_description = [role1, role2, role3, role4, role5]
    workflow_description = " -> ".join(workflow_description)

    print("======= Workflow Testing =======\n")
    
    print("Initializing workflow...")
    print(f"\nTask name: {task_name}")
    print(f"Workflow description: {workflow_description}")

    agents = []
    agents.append(Agent(role=role1, workflow_description=workflow_description))
    agents.append(Agent(role=role2, workflow_description=workflow_description))
    agents.append(Agent(role=role3, workflow_description=workflow_description))
    agents.append(Agent(role=role4, workflow_description=workflow_description))
    agents.append(Agent(role=role5, workflow_description=workflow_description))
    workflow = Workflow(task_name=task_name, agents=agents)

    print("\nInput data:")
    print(input_data)
    print("\nRunning workflow...")

    result = workflow.run(input_data)
    print("\nResponse:")
    print(result['content'])
    print(f"\nPrompt tokens: {result['prompt_tokens']}")
    print(f"Response tokens: {result['response_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
