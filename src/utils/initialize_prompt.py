# src/utils/initialize_prompt.py
import re
import yaml
from pathlib import Path

from src.config import DEFAULT_PROMPTS, TASK_DESCRIPTIONS, ROLE_DESCRIPTIONS
from src.llm.client import PromptGenerator

PROJECT_ROOT = Path(__file__).parent.parent.parent
default_prompt = DEFAULT_PROMPTS.get('DefaultAgentGenerationPrompt', {}).get('prompt', '')

def parse_prompt(raw_prompt: str) -> str:
    """
    Extract content between <prompt> and </prompt> tags.
    """
    pattern = r'<prompt>(.*?)</prompt>'
    match = re.search(pattern, raw_prompt, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return raw_prompt.strip()

def initialize_prompt():
    """
    Generate prompts for each task-role combination
    """
    prompt_generator = PromptGenerator()
    generated_prompts = {}

    for task_name, task_info in TASK_DESCRIPTIONS.items():
        task_description = task_info.get('description', '')
        if not task_description:
            task_description = f"This task is for {task_name}."
        
        generated_prompts[task_name] = {}

        for role in ROLE_DESCRIPTIONS:
            role_name = role.get('name', '')
            role_description = role.get('description', '')
            role_category = role.get('category', '')

            description = (
                f"Task: {task_name}\n"
                f"Task Description: {task_description}\n\n"
                f"Role: {role_name}\n"
                f"Role Description: {role_description}\n"
                f"Role Category: {role_category}"
            )
            
            raw_prompt = prompt_generator.generate_prompt(
                default_prompt=default_prompt,
                description=description
            )
            prompt = parse_prompt(raw_prompt)

            generated_prompts[task_name][role_name] = prompt
            print(f"Generated prompt for Task: {task_name}, Role: {role_name}")
    
    print(f"Total prompts generated: {sum(len(roles) for roles in generated_prompts.values())}")
    return generated_prompts

def save_prompts(prompts: dict, filename: str = 'initial_prompts.yaml'):
    """
    Save generated prompts to a YAML file
    """
    output_path = PROJECT_ROOT / 'configs' / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            prompts,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )
    print(f"Prompts saved to: {output_path}")
    return


if __name__ == "__main__":
    prompts = initialize_prompt()
    save_prompts(prompts)
