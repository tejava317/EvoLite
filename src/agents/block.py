from src.agents.agent import Agent
import re

# =============================
# Block type.
# Freely add the blocks.
# =============================
# Note: Please implement following APIs for each block.
# 1) expand(self, input_text="") : expand Block class to Agent class
# 2) copy(self) : copy itself
# 3) __str__(self)
# 4) attribute self.num_agents

class Block:

    def expand(self, input_text):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
    
    def __str__(self):
        raise NotImplementedError()

# AgentBlock class.
# It implies a simple agent.
class AgentBlock(Block):

    def __init__(self, role):
        self.role = role
        self.num_agents = 1

    def expand(self, workflow_description=""):
        return [Agent(role=self.role, workflow_description=workflow_description)]
    
    def copy(self):
        return AgentBlock(self.role)
    
    def __str__(self):
        return self.role

# CompositeBlock class.
# It implies a divider and synthesizer format.
class CompositeBlock(Block):

    def __init__(self, divider_role="Divider", synth_role="Synthesizer"):
        self.divider_role = divider_role
        self.synth_role = synth_role
        self.num_agents = 2
        self.agents = None

    def _parse_inner_roles(self, text):

        if ":" in text:
            text = text.split(":", 1)[1]
        text = text.replace("\n", ",").replace("•", ",")

        parts = re.split(r"[,;\-\d\.\)]+", text)
        roles = [p.strip() for p in parts if p.strip()]
        return roles

    def expand(self, workflow_description):

        num_inner_roles = 3

        # Divider
        divider = Agent(role=self.divider_role, workflow_description=workflow_description)
        divider_prompt = f"List {num_inner_roles} roles to make an efficient work." # Have to decide
        divider.prompt = divider_prompt
        # Run the divider to decide the inner roles.
        try:
            div_output = divider.run(divider_prompt)["content"].strip()
            if not div_output:
                div_output = default_role
        except Exception:
            div_output = "Business Analyst, Install KaKao Talk, Snack Eating"

        inner_roles = self._parse_inner_roles(div_output)

        agents = [Agent(role=self.divider_role, workflow_description=workflow_description)]
        agents += [Agent(role=r, workflow_description=workflow_description) for r in inner_roles]
        
        # Synthesizer
        synthesizer = Agent(role=self.synth_role, workflow_description=workflow_description)
        synthesizer.prompt = f"Synthesize {num_inner_roles} roles to synthesize." # Have to decide
        agents.append(synthesizer)

        self.num_agents = num_inner_roles + 2
        self.agents = agents

        return agents
    
    def copy(self):
        return CompositeBlock(
            divider_role=self.divider_role,
            synth_role=self.synth_role
        )
    
    def __str__(self):

        string = ""
        if self.agents == None:
            string = f"Divider -> Synthesizer"
        else:
            string_list = [f"{str(agent)}" for agent in self.agents]
            string = "(" + " , ".join(string_list) + ")"
        return string
