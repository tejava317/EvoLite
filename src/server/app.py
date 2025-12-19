# src/server/app.py
"""
FastAPI application setup and lifespan management.

Supports two modes:
- vllm: Uses local vLLM server (Qwen, etc.)
- openai: Uses OpenAI API (GPT-4o-mini, etc.)

Set LLM_MODE environment variable to switch:
    export LLM_MODE=openai  # or vllm (default)

Run with:
    uvicorn src.server.app:app --host 0.0.0.0 --port 8001
"""

import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from langchain_openai import ChatOpenAI

from .state import state
from .endpoints import router
from ..schemas import TASK_SCHEMAS, ExtractorResponse
from ..datasets import MBPPDataset, MathAlgebraDataset, CRUXOpenDataset

# Load environment variables from .env file
load_dotenv()


def create_llm():
    """Create LLM based on LLM_MODE environment variable."""
    llm_mode = os.getenv("LLM_MODE", "vllm").lower()
    
    if llm_mode == "openai":
        # OpenAI mode - uses OpenAI API
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set! Required for openai mode.")
        
        if model == "gpt-5-nano":
            llm = ChatOpenAI(
                model="gpt-5-nano",
                max_completion_tokens=4096,
                reasoning={"effort": "minimal"},
                request_timeout=120.0,
            )
        else:
            llm = ChatOpenAI(
                model=model,
                temperature=0.6,
                max_tokens=4096,
                request_timeout=120.0,
            )

        # llm = ChatOpenAI(
        #     model=model,
        #     temperature=0.6,
        #     max_tokens=4096,
        #     request_timeout=120.0,
        # )
        
        print(f"  Mode: OpenAI")
        print(f"  Model: {model}")
        return llm, "openai", model
    
    else:
        # vLLM mode (default) - uses local vLLM server
        base_url = os.getenv("VLLM_BASE_URL", "http://103.196.86.181:28668/v1")
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B")
        
        llm = ChatOpenAI(
            model=model,
            openai_api_key="EMPTY",  # vLLM doesn't require a real key
            openai_api_base=base_url,
            temperature=0.6,
            max_tokens=2048,
            request_timeout=600.0,
        )
        
        print(f"  Mode: vLLM")
        print(f"  URL: {base_url}")
        print(f"  Model: {model}")
        return llm, "vllm", model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources at startup, cleanup at shutdown."""
    print("🚀 Starting Evaluation Server (LangChain Edition)...")
    
    # Initialize LLM
    state.llm, state.llm_mode, state.llm_model = create_llm()
    print(f"✓ LangChain ChatOpenAI initialized")
    
    # Pre-create structured output models for each task
    state.structured_llms = {}
    for task_name, schema in TASK_SCHEMAS.items():
        state.structured_llms[task_name] = state.llm.with_structured_output(schema)
        print(f"  ✓ Structured output ready for {task_name}")
    
    # Extractor model (simpler schema)
    state.structured_llms["EXTRACTOR"] = state.llm.with_structured_output(ExtractorResponse)
    
    # Pre-load datasets
    print("📂 Loading datasets...")
    state.datasets = {}
    
    try:
        mbpp = MBPPDataset(split="test")
        mbpp.load()
        state.datasets["MBPP"] = mbpp
        print(f"  ✓ MBPP: {len(mbpp)} problems")
    except Exception as e:
        print(f"  ✗ MBPP failed: {e}")
    
    try:
        math_ds = MathAlgebraDataset(split="test")
        math_ds.load()
        state.datasets["MATH"] = math_ds
        print(f"  ✓ MATH: {len(math_ds)} problems")
    except Exception as e:
        print(f"  ✗ MATH failed: {e}")

    try:
        crux = CRUXOpenDataset(split="test")
        crux.load()
        state.datasets["CRUX-O"] = crux
        print(f"  ✓ CRUX-O: {len(crux)} problems")
    except Exception as e:
        print(f"  ✗ CRUX-O failed: {e}")
    
    print(f"✓ Server ready on port 8001! (Mode: {state.llm_mode})")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")


# Silence access logs for noisy endpoints
class _EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/stats " not in msg


logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())


# Create the FastAPI app
app = FastAPI(
    title="EvoLite Evaluation Server",
    description="High-throughput workflow evaluation using LangChain + vLLM/OpenAI with structured output",
    version="4.1.0",
    lifespan=lifespan
)

# Include the router
app.include_router(router)


# ============== BlockWorkflow Recreation ==============

def create_block_workflow_from_config(config):
    """Recreate a BlockWorkflow object from a WorkflowConfig."""
    from ..agents.block import AgentBlock, CompositeBlock
    from ..agents.workflow_block import BlockWorkflow
    
    blocks = []
    for block_config in config.blocks:
        if block_config.type == "agent":
            blocks.append(AgentBlock(role=block_config.role))
        elif block_config.type == "composite":
            blocks.append(CompositeBlock(
                divider_role=block_config.divider_role,
                synth_role=block_config.synth_role
            ))
    
    return BlockWorkflow(task_name=config.task_name, blocks=blocks)


def workflow_to_config(workflow):
    """Convert a BlockWorkflow to WorkflowConfig for API calls."""
    from ..agents.block import AgentBlock, CompositeBlock
    from .models import BlockConfig, WorkflowConfig
    
    blocks = []
    for block in workflow.blocks:
        if isinstance(block, AgentBlock):
            blocks.append(BlockConfig(type="agent", role=block.role))
        elif isinstance(block, CompositeBlock):
            blocks.append(BlockConfig(
                type="composite",
                divider_role=block.divider_role,
                synth_role=block.synth_role
            ))
    
    return WorkflowConfig(
        blocks=blocks,
        task_name=workflow.task_name,
        use_extractor=True
    )


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
