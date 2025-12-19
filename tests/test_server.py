import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import EvaluationClient, BlockConfig

async def test():
    client = EvaluationClient('http://localhost:8001')
    blocks = [BlockConfig(type="agent", role="Solution Drafter")]
    result = await client.evaluate_batch_async([blocks], 'MBPP', num_problems=1)
    print(f'Result: {result}')
    if result:
        print(f'Type: {type(result[0])}')
        print(f'Attributes: {dir(result[0])}')
        print(f'pass_at_1: {result[0].pass_at_1}')
        print(f'total_tokens: {result[0].total_tokens}')
        if hasattr(result[0], '__dict__'):
            print(f'Dict: {result[0].__dict__}')
    await client.close()

asyncio.run(test())

