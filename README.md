# EvoLite

Evolving Multi-Agent Workflow for Lite-LLM


## Environment Setup

Follow these steps to set up the development environment.

### 1. Clone the Repository
```bash
git clone https://github.com/tejava317/EvoLite.git
cd evolite
```

### 2. Create Conda environment and install requirements

Use the following commands to create the conda environment with all required packages.
```bash
conda env create -f environment.yml
conda activate evolite
```

### 3. Set Up Environment Variables

Copy the example environment file.
```bash
cp .env.example .env
```

Open the `.env` file and enter your API key.
```
OPENAI_API_KEY=your_openai_api_key
```
