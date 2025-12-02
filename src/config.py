# src/config.py
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration - supports both OpenAI and RunPod
# Priority: RunPod > OpenAI

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Determine which API to use
USE_RUNPOD = bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)

if USE_RUNPOD:
    API_KEY = RUNPOD_API_KEY
    BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
    print(f"Using RunPod API (endpoint: {RUNPOD_ENDPOINT_ID})")
elif OPENAI_API_KEY:
    API_KEY = OPENAI_API_KEY
    BASE_URL = None  # Use default OpenAI URL
    print("Using OpenAI API")
else:
    raise ValueError(
        "No API key configured. Set either:\n"
        "  - RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID for RunPod, or\n"
        "  - OPENAI_API_KEY for OpenAI"
    )

# Load default prompts.
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "configs" / "default_prompts.yaml"

def load_default_prompts():
    """
    Load default prompts from YAML configuration file
    """
    if not DEFAULT_PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Default prompts file not found: {DEFAULT_PROMPTS_PATH}")
    
    with open(DEFAULT_PROMPTS_PATH, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    
    return prompts

DEFAULT_PROMPTS = load_default_prompts()


# Load task descriptions.
TASK_DESCRIPTIONS_PATH = PROJECT_ROOT / "configs" / "task_descriptions.yaml"

def load_task_descriptions():
    """
    Load task descriptions from YAML configuration file
    """
    if not TASK_DESCRIPTIONS_PATH.exists():
        raise FileNotFoundError(f"Task descriptions file not found: {TASK_DESCRIPTIONS_PATH}")
    
    with open(TASK_DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
        task_descriptions = yaml.safe_load(f)
    
    return task_descriptions

TASK_DESCRIPTIONS = load_task_descriptions()

# Role description 
ROLE_DESCRIPTIONS_PATH = PROJECT_ROOT / "configs" / "role.yaml"

def load_role_descriptions():
    if not ROLE_DESCRIPTIONS_PATH.exists():
        raise FileNotFoundError(f"Role descriptions file not found: {ROLE_DESCRIPTIONS_PATH}")
    
    with open(ROLE_DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
        role_descriptions = yaml.safe_load(f)
    
    return role_descriptions

ROLE_DESCRIPTIONS = load_role_descriptions()


# Base agents with predefined prompts
BASE_AGENTS_PATH = PROJECT_ROOT / "configs" / "base_agents.yaml"

def load_base_agents():
    """Load predefined agent prompts from base_agents.yaml"""
    if not BASE_AGENTS_PATH.exists():
        return {}
    
    with open(BASE_AGENTS_PATH, 'r', encoding='utf-8') as f:
        base_agents = yaml.safe_load(f)
    
    return base_agents or {}

BASE_AGENTS = load_base_agents()


def get_predefined_prompt(role: str) -> str:
    """
    Get predefined prompt for a role from base_agents.yaml.
    
    Matches by role name (case-insensitive, flexible matching).
    Returns None if no predefined prompt exists.
    """
    role_lower = role.lower().replace(" ", "").replace("_", "")
    
    for agent_key, agent_config in BASE_AGENTS.items():
        # Match by task name or key
        task = agent_config.get("task", "").lower().replace(" ", "").replace("_", "")
        key_normalized = agent_key.lower().replace(" ", "").replace("_", "")
        
        if role_lower == task or role_lower == key_normalized:
            return agent_config.get("prompt", "")
    
    return None
