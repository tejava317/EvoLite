# src/config.py
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set in the .env file.")

# ANTROPIC_API_KEY = os.getenv("ANTROPIC_API_KEY")

# if not ANTROPIC_API_KEY:
#     raise ValueError("Anthropic API key is not set in the .env file.")

# Load defualt prompts.
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