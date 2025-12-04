# src/agents/workflow_block.py
from typing import TypedDict, List, Annotated, Optional
import operator
import re
from langgraph.graph import StateGraph, END
from src.config import TASK_DESCRIPTIONS, DEFAULT_PROMPTS
from src.agents.agent import Agent
from src.agents.block import Block, AgentBlock, CompositeBlock
from src.llm.client import PromptGenerator


class WorkflowState(TypedDict):
    
    input: str
    current_output: str
    intermediate_results: Annotated[List[str], operator.add]
    prompt_tokens: Annotated[int, operator.add]
    response_tokens: Annotated[int, operator.add]
    total_tokens: Annotated[int, operator.add]

# =============================
# BlockWorkflow class.
# Workflow composed of blocks.
# =============================
class BlockWorkflow:
    def __init__(self,
                 task_name: str,
                 blocks: Optional[List[Block]] = None):
        
        # Agent and task information
        self.task_name = task_name
        self.blocks: List[Block] = blocks if blocks is not None else [] # Block expression
        self.agents = None  # Agent expression

        self.num_agents = None

        # Graph information
        self.workflow_graph = None
        self.compiled_graph = None

        # evaluation
        self.response = None
        self.prompt_tokens = None
        self.response_tokens = None
        self.total_tokens = None

    # Expand blocks to agents.
    # After expand, revise the workflow description and build a graph.
    def _expand_blocks_to_agents(self):
        agents = []

        # Extend block to the agent.
        for block in self.blocks:
            agents.extend(block.expand(""))

        # Generate the task description.
        description = " -> ".join(agent.role for agent in agents)
        for agent in agents:
            agent.workflow_description = description

        self.agents = agents
        self._build_graph()

    # Initialize the prompt.
    def initialize_prompt(self):
    
        default_prompt = DEFAULT_PROMPTS.get('DefaultWorkflowGenerationPrompt', {}).get('prompt', '')

        if not default_prompt:
            raise ValueError("DefaultWorkflowGenerationPrompt not found in configuration")

        task_description = TASK_DESCRIPTIONS.get(self.task_name, {}).get('description', '')

        if not task_description:
            task_description = f"This workflow is for {self.task_name} task."

        default_prompt = default_prompt.replace('[Dataset description]', task_description)
        return default_prompt

    # Insert a block at the specific poistion.
    # Aware that the new agent's role becomes an argument for an ease of implementation.
    def insert_block(self, block: Block, position: int):

        # Position error handling.
        if position < 0 or position > len(self.blocks):
            raise IndexError("Position out of range while inserting block.")
        self.blocks.insert(position, block)

        # Clear expanded graph state
        self.agents = []
        self.workflow_graph = None
        self.compiled_graph = None

    # Remove a block at the specif position.
    def remove_block(self, position: int):
        
        # Position error handling.
        if position < 0 or position >= len(self.blocks):
            raise IndexError("Position out of range while removing block.")
        removed = self.blocks.pop(position)

        # Clear expanded state
        self.agents = []
        self.workflow_graph = None
        self.compiled_graph = None
        return removed

    # Create the agent node.
    # Handler the divider block differently.
    def _create_agent_node(self, agent: Agent, agent_idx: int):
        def agent_node(state: WorkflowState) -> WorkflowState:
            input_data = state["current_output"] if state.get("current_output") else state["input"]
            
            if agent.role == "divider":
                modified_prompt = f"""
                You are now functioning as a DIVIDER agent.

                Your job:
                1. Read the previous result.
                2. Divide the result with 3 roles.

                Previous output:
                {input_data}

                Return ONLY the rewritten prompt.
                """
                input_data = modified_prompt
            
            response = agent.run(input_data)
            return {
                "current_output": response['content'],
                "intermediate_results": [f"Agent {agent_idx} ({agent.role}): {response['content'][:100]}..."],
                "prompt_tokens": response.get('prompt_tokens', 0),
                "response_tokens": response.get('response_tokens', 0),
                "total_tokens": response.get('total_tokens', 0)
            }
        return agent_node

    # Build a graph.
    # Be sure that self.agents is made.
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

    # Run the graph.
    def run(self, input_data: str) -> dict:
        
        # Expand blocks.
        if self.blocks:
            self._expand_blocks_to_agents()

        if not self.compiled_graph:
            raise ValueError("Workflow has no agents. Add agents or blocks before running the workflow.")

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
        agent_names = [f"{i+1}. {block}" for i, block in enumerate(self.blocks)]
        return f"Workflow for {self.task_name}:\n" + "\n".join(agent_names)

    def copy(self):
        copied_blocks = [block.copy() for block in self.blocks]
        return BlockWorkflow(self.task_name, blocks=copied_blocks)


if __name__ == "__main__":
    task_name = "HumanEval"
    input_data = "Write a Python function that calculates the factorial of a number."

    # ===== 기존 Agent 기반 예시를 Block 기반으로 변환 =====
    role1 = "Task Parsing Agent"
    role2 = "Task Refinement Agent"
    role3 = "Code Generation Agent"
    role4 = "Code Reviewer Agent"
    role5 = "Code Refinement Agent"

    # 각 역할을 단일 AgentBlock으로 감싸기
    blocks: List[Block] = [
        AgentBlock(role1),
        AgentBlock(role2),
        AgentBlock(role3),
        AgentBlock(role4),
        CompositeBlock(),
    ]

    print("======= Workflow Testing (Block-based) =======\n")

    print("Initializing workflow with Blocks...")
    print(f"\nTask name: {task_name}")
    print("Blocks:")
    for i, b in enumerate(blocks, 1):
        if isinstance(b, AgentBlock):
            print(f"  {i}. AgentBlock(role={b.role})")
        elif isinstance(b, CompositeBlock):
            print(f"  {i}. CompositeBlock(divider={b.divider_role}, synth={b.synth_role})")

    workflow = Workflow(task_name=task_name, blocks=blocks)

    print("\nInput data:")
    print(input_data)
    print("\nRunning workflow...")

    # result = workflow.run(input_data)
    # print("\nResponse:")
    # print(result['content'])
    # print(f"\nPrompt tokens: {result['prompt_tokens']}")
    # print(f"Response tokens: {result['response_tokens']}")
    # print(f"Total tokens: {result['total_tokens']}")
