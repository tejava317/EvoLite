# EvoLite

**Evolutionary Multi-Agent Workflow Optimization for LLMs**

EvoLite is a framework that uses evolutionary algorithms to automatically discover and optimize multi-agent workflows for code generation tasks. It balances **performance (Pass@1)** and **efficiency (token cost)** using multi-objective optimization (NSGA-II).

## Key Features

- 🧬 **Evolutionary Workflow Optimization**: Uses genetic algorithms to evolve agent workflow topologies
- 🎯 **Multi-Objective Optimization**: NSGA-II based Pareto optimization for Pass@1 vs Token Cost trade-off
- 🤖 **LLM-Driven Evolution**: Semantic mutation and crossover using LLM for intelligent topology modifications
- 📊 **Benchmark Support**: MBPP and MATH algebra benchmarks for evaluation
- 🔄 **Flexible Workflow Design**: Linear chains, loops (reflexion), branching (plan-and-solve), and test-driven patterns
- ⚡ **High-Performance Evaluation**: Async batch evaluation with configurable concurrency

## Architecture

```
EvoLite/
├── src/
│   ├── agents/          # Agent and Workflow definitions
│   │   ├── agent.py     # Base Agent class
│   │   ├── workflow.py  # Workflow orchestration using LangGraph
│   │   ├── block.py     # Block-based agent abstraction
│   │   └── extractors.py # Answer extraction agents
│   ├── ga/              # Genetic Algorithm implementations
│   │   ├── ga_llm.py    # LLM-enhanced GA (main algorithm)
│   │   ├── ga.py        # Basic GA implementation
│   │   ├── hdlo.py      # Hierarchical Design-Level Optimization
│   │   └── multi_objective.py  # NSGA-II utilities
│   ├── datasets/        # Benchmark dataset loaders
│   │   ├── mbpp.py      # MBPP dataset
│   │   └── math_algebra.py  # MATH algebra dataset
│   ├── evaluation/      # Evaluation utilities
│   ├── llm/             # LLM client implementations
│   ├── client/          # Evaluation server client
│   └── server/          # FastAPI evaluation server
├── configs/             # Configuration files
├── scripts/             # Baseline scripts
│   ├── mbpp_baseline.py # MBPP baseline solver
│   └── math_baseline.py # MATH baseline solver
├── tests/               # Test scripts
└── evaluate.py          # Single workflow evaluation
```

## How It Works

### 1. Workflow Representation
Workflows are represented as directed graphs using arrow syntax:
```
Planner -> Coder -> Reviewer -> Coder  # Reflexion loop
Planner -> CoderA, Planner -> CoderB -> Merger  # Branching
```

### 2. Evolutionary Operators
- **Mutation**: Add/remove agents, rewire connections
  - Semantic mutation: LLM suggests improvements
  - Agnostic mutation: Random structural changes
- **Crossover**: Combine topologies from two parent workflows
  - Distillation: Transplant modules between workflows
  - Mixing: Merge parallel structures

### 3. Fitness Evaluation
Each workflow is evaluated on:
- **Pass@1**: Code correctness on benchmark problems
- **Token Cost**: Total tokens used for inference

### 4. Selection (NSGA-II)
- Non-dominated sorting to identify Pareto fronts
- Crowding distance for diversity preservation
- Elitism with buffer (probation) mechanism

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tejava317/EvoLite.git
cd EvoLite
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate evolite
```

### 3. Set Up Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and configure your API:
```bash
# Option 1: OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Option 2: Local vLLM server (OpenAI-compatible)
VLLM_BASE_URL=http://localhost:8000/v1
```

## Usage

### Run Evolutionary Optimization

```bash
# Run GA-LLM on MBPP dataset
python -m src.ga.ga_llm \
    --task MBPP \
    --population-size 100 \
    --generation 15 \
    --num-problem 30 \
    --server-url http://localhost:8001

# Run without LLM (random mutation only)
python -m src.ga.ga_llm --task MBPP --no-llm --fast
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--population-size` | 100 | Number of workflows per generation |
| `--generation` | 10 | Number of evolutionary generations |
| `--num-problem` | 30 | Problems for fitness evaluation |
| `--elite-ratio` | 0.2 | Fraction of population preserved |
| `--buffer-size` | 10 | Probation pool size |
| `--max-eval-iter` | 4 | Max evaluations per individual |
| `--no-llm` | - | Disable LLM-based operators |
| `--fast` | - | Use random fitness (for testing) |

### Evaluate a Single Workflow

```bash
# Evaluate default workflow on MBPP
python evaluate.py --task MBPP --num-problems 50

# Custom workflow
python evaluate.py --task MBPP \
    --roles "Task Parsing Agent,Code Generation Agent,Code Reviewer Agent" \
    --show-intermediate
```

### Run Baseline (No Multi-Agent)

```bash
# Single-agent baseline on MBPP
python scripts/mbpp_baseline.py --num_problems 100 --model "Qwen/Qwen3-4B"
```

### Start Evaluation Server

```bash
# Start FastAPI server for batch evaluation
uvicorn src.server.app:app --host 0.0.0.0 --port 8001
```

## Output

### Checkpoints
Evolution checkpoints are saved to `src/ga/ga_llm_checkpoints/`:
```
population_<run_id>_gen0.csv
population_<run_id>_gen1.csv
...
population_<run_id>_final.csv
```

### Pareto Plots
Visualization of Pareto fronts saved to `src/ga/ga_llm_graph/`:
```
<run_id>_gen0.png
<run_id>_gen1.png
...
<run_id>_final.png
```

## Configuration

### Agent Roles
Define custom agent roles in `configs/generated_prompts.yaml`:
```yaml
agents:
  Task Parsing Agent:
    prompt: "You are a task parsing agent..."
  Code Generation Agent:
    prompt: "You are a code generation agent..."
```

### Task Descriptions
Configure benchmark tasks in `configs/task_descriptions.yaml`:
```yaml
MBPP:
  description: "Python programming problems from MBPP..."
MATH:
  description: "Algebra problems from MATH dataset..."
```

## Algorithm Details

### GA-LLM Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `POPULATION_SIZE` | 100 | Individuals per generation |
| `MUTATION_RATE` | 0.7 | Probability of mutation |
| `CROSSOVER_RATE` | 0.3 | Probability of crossover |
| `AGNOSTIC_RATIO` | 0.2 | Random vs semantic mutation |
| `LLM_CALL_BUDGET` | 500 | Max LLM calls for evolution |

### Workflow Patterns
1. **Linear Chain**: `A -> B -> C`
2. **Reflexion Loop**: `Solver -> Reviewer -> Solver`
3. **Branching**: `Planner -> [WorkerA, WorkerB] -> Merger`
4. **Test-Driven**: `TestGen -> CodeGen -> Verify`

## License

MIT License

## Citation

If you use EvoLite in your research, please cite:
```bibtex
@software{evolite2024,
  title = {EvoLite: Evolutionary Multi-Agent Workflow Optimization},
  author = {EvoLite Team},
  year = {2025},
  url = {https://github.com/tejava317/EvoLite}
}
```
