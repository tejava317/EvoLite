# EvoLite Evaluation System Guide

A high-throughput evaluation system for multi-agent workflows using RunPod's async API.

## Quick Start

### 1. Environment Setup

```bash
# Activate the conda environment
conda activate evolite

# Set up environment variables in .env file
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

### 2. Start the Evaluation Server

```bash
cd EvoLite
python -m uvicorn src.evaluation_server:app --host 0.0.0.0 --port 8000
```

### 3. Run Evaluation

```python
from src.evaluation_client import EvaluationClient

client = EvaluationClient("http://localhost:8000")

# Simple evaluation with role names
result = client.evaluate_simple(
    roles=["Code Generation Agent"],
    task_name="MBPP",
    num_problems=10
)

print(f"Pass@1: {result.pass_at_1:.2%}")
print(f"Correct: {result.num_correct}/{result.num_problems}")
print(f"Tokens: {result.total_tokens}")
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Simple Evaluation (Role List)
```bash
curl -X POST http://localhost:8000/evaluate/simple \
  -H "Content-Type: application/json" \
  -d '{
    "roles": ["Code Generation Agent"],
    "task_name": "MBPP",
    "num_problems": 5,
    "use_extractor": true
  }'
```

### Block-based Evaluation
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "blocks": [
        {"type": "agent", "role": "Task Parsing Agent"},
        {"type": "agent", "role": "Code Generation Agent"}
      ],
      "task_name": "MBPP",
      "use_extractor": true
    },
    "num_problems": 5
  }'
```

### Quick Evaluation (URL Parameters)
```bash
curl "http://localhost:8000/evaluate/quick?roles=Code%20Generation%20Agent&task=MBPP&num_problems=3"
```

### Batch Evaluation (Compare Workflows)
```bash
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "workflows": [
      {"blocks": [{"type": "agent", "role": "Code Generation Agent"}], "task_name": "MBPP"},
      {"blocks": [{"type": "agent", "role": "Task Parsing Agent"}, {"type": "agent", "role": "Code Generation Agent"}], "task_name": "MBPP"}
    ],
    "num_problems": 5,
    "seed": 42
  }'
```

---

## Python Client Usage

### Basic Usage

```python
from src.evaluation_client import EvaluationClient, BlockConfig

client = EvaluationClient("http://localhost:8000")

# Check server health
health = client.health_check()
print(health)  # {"status": "healthy", "datasets": ["MBPP", "MATH"], ...}
```

### Simple Role-based Evaluation

```python
# Single agent
result = client.evaluate_simple(
    roles=["Code Generation Agent"],
    task_name="MBPP",
    num_problems=10,
    use_extractor=True,
    seed=42  # For reproducibility
)

# Multi-agent pipeline
result = client.evaluate_simple(
    roles=["Task Parsing Agent", "Code Generation Agent", "Code Reviewer Agent"],
    task_name="MBPP",
    num_problems=10
)
```

### Block-based Evaluation

```python
from src.evaluation_client import BlockConfig

# Create blocks
blocks = [
    BlockConfig(type="agent", role="Task Parsing Agent"),
    BlockConfig(type="agent", role="Code Generation Agent"),
]

result = client.evaluate(
    blocks=blocks,
    task_name="MBPP",
    num_problems=10
)
```

### Composite Blocks (Divider → Inner Agents → Synthesizer)

```python
blocks = [
    BlockConfig(type="agent", role="Task Parsing Agent"),
    BlockConfig(
        type="composite",
        divider_role="Divider",
        synth_role="Synthesizer"
    ),
]

result = client.evaluate(blocks=blocks, task_name="MBPP", num_problems=5)
```

### Batch Comparison

```python
# Compare multiple workflows on the same problems
results = client.evaluate_batch_simple(
    workflows=[
        ["Code Generation Agent"],
        ["Task Parsing Agent", "Code Generation Agent"],
        ["Task Parsing Agent", "Code Generation Agent", "Code Reviewer Agent"],
    ],
    task_name="MBPP",
    num_problems=10,
    seed=42  # Same problems for fair comparison
)

for i, r in enumerate(results):
    print(f"Workflow {i+1}: Pass@1={r.pass_at_1:.2%}, Tokens={r.total_tokens}")
```

### Async Usage

```python
import asyncio
from src.evaluation_client import EvaluationClient

async def main():
    client = EvaluationClient("http://localhost:8000")
    
    result = await client.evaluate_simple_async(
        roles=["Code Generation Agent"],
        task_name="MBPP",
        num_problems=10
    )
    
    print(f"Pass@1: {result.pass_at_1:.2%}")
    await client.close()

asyncio.run(main())
```

---

## Available Datasets

| Dataset | Task Type | Problems | Description |
|---------|-----------|----------|-------------|
| MBPP | Code Generation | 257 | Python programming problems |
| MATH | Math (Algebra) | 1187 | Algebra problems with LaTeX answers |

---

## Available Agent Roles

The system includes 54 predefined roles with task-specific prompts:

**Core Development:**
- Code Generation Agent
- Code Reviewer Agent
- Code Refinement Agent
- Code Testing Agent
- Unit Tester

**Planning & Analysis:**
- Task Parsing Agent
- Task Refinement Agent
- Business Analyst
- Requirement Engineer
- Software Architect

**Quality & Security:**
- Security Engineer
- Performance Engineer
- Quality Assurance Tester
- DevOps Engineer

**Specialized:**
- Algorithm Specialist
- Data Scientist
- Machine Learning Engineer
- API Developer
- Database Administrator

See `configs/role.yaml` for the full list.

---

## Task-Specific Prompts

The system uses pre-generated prompts from `configs/initial_prompts.yaml`:

```python
from src.config import get_predefined_prompt

# Get task-specific prompt
prompt = get_predefined_prompt("Code Generation Agent", "MBPP")

# Get generic prompt (fallback)
prompt = get_predefined_prompt("Code Generation Agent")
```

---

## Response Format

```python
@dataclass
class EvalResult:
    pass_at_1: float        # Success rate (0.0 to 1.0)
    num_correct: int        # Number of correct solutions
    num_problems: int       # Total problems evaluated
    total_tokens: int       # Total tokens used
    total_time: float       # Total execution time (seconds)
    tokens_per_second: float # Throughput
    error: Optional[str]    # Error message if failed
```

---

## Integration with Genetic Algorithm

```python
from src.evaluation_client import evaluate_block_workflow
from src.agents.workflow_block import BlockWorkflow
from src.agents.block import AgentBlock

# Create a BlockWorkflow
workflow = BlockWorkflow(
    task_name="MBPP",
    blocks=[
        AgentBlock("Task Parsing Agent"),
        AgentBlock("Code Generation Agent"),
    ]
)

# Evaluate via server
fitness = evaluate_block_workflow(
    workflow=workflow,
    num_problems=5,
    server_url="http://localhost:8000",
    token_penalty=0.0001
)

print(f"Fitness: {fitness}")
```

### Running GA with Server Evaluation

```bash
cd EvoLite
python -m src.ga.ga --task MBPP --server --server-url http://localhost:8000
```

---

## Generating New Prompts

To generate prompts for new task-role combinations:

```bash
cd EvoLite
python -m src.utils.initialize_prompt
```

This will regenerate `configs/initial_prompts.yaml` with prompts for all task-role combinations.

---

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :8000

# Kill existing process
pkill -f "uvicorn src.evaluation_server"
```

### Connection refused
```bash
# Verify server is running
curl http://localhost:8000/health

# Check server logs
python -m uvicorn src.evaluation_server:app --host 0.0.0.0 --port 8000 --log-level debug
```

### RunPod API errors
- Verify `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in `.env`
- Check RunPod dashboard for endpoint status
- Ensure endpoint is running and not scaled to zero

### Low pass rates
- Try different agent combinations
- Increase `num_problems` for more reliable estimates
- Use `seed` parameter for reproducible results

---

## Example: Full Evaluation Script

```python
#!/usr/bin/env python
"""Example: Evaluate multiple workflows and compare results."""

from src.evaluation_client import EvaluationClient

def main():
    client = EvaluationClient("http://localhost:8000")
    
    # Check server
    health = client.health_check()
    print(f"Server status: {health['status']}")
    print(f"Available datasets: {health['datasets']}")
    
    # Define workflows to compare
    workflows = {
        "Single Agent": ["Code Generation Agent"],
        "Two Agents": ["Task Parsing Agent", "Code Generation Agent"],
        "Three Agents": ["Task Parsing Agent", "Code Generation Agent", "Code Reviewer Agent"],
    }
    
    # Evaluate each workflow
    print("\n" + "=" * 60)
    print("MBPP Evaluation Results")
    print("=" * 60)
    
    for name, roles in workflows.items():
        result = client.evaluate_simple(
            roles=roles,
            task_name="MBPP",
            num_problems=20,
            seed=42
        )
        
        print(f"\n{name}:")
        print(f"  Roles: {' → '.join(roles)}")
        print(f"  Pass@1: {result.pass_at_1:.2%}")
        print(f"  Correct: {result.num_correct}/{result.num_problems}")
        print(f"  Tokens: {result.total_tokens}")
        print(f"  Time: {result.total_time:.1f}s")
        print(f"  Throughput: {result.tokens_per_second:.1f} tokens/s")

if __name__ == "__main__":
    main()
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Evaluation Client                      │
│  (evaluate_simple, evaluate, evaluate_batch)            │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/JSON
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 FastAPI Server (:8000)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ /evaluate   │  │ /evaluate/  │  │ /evaluate/  │     │
│  │ /simple     │  │ /batch      │  │ /quick      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                         │                               │
│  ┌──────────────────────▼──────────────────────┐       │
│  │           RunPodAsyncClient                  │       │
│  │  (Fire-all-at-once async API calls)         │       │
│  └──────────────────────┬──────────────────────┘       │
│                         │                               │
│  ┌──────────────────────▼──────────────────────┐       │
│  │         Datasets (MBPP, MATH)               │       │
│  │         Prompts (initial_prompts.yaml)      │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼ RunPod Native API
┌─────────────────────────────────────────────────────────┐
│                   RunPod Serverless                      │
│                   (vLLM Endpoint)                        │
└─────────────────────────────────────────────────────────┘
```

