#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evolite

cd /root/project/EvoLite
export PYTHONUNBUFFERED=1

nohup python -u -m src.ga.ga_llm \
    --task MATH \
    --run-id math_ga_llm_v1 \
    --server-url http://localhost:8002 \
    --population-size 20 \
    --generation 15 \
    --num-problem 20 \
    --elite-ratio 0.5 \
    --buffer-size 10 \
    --max-eval-iter 4 \
    --finalize-valid 100 \
    > /root/project/EvoLite/src/ga/ga_llm_math.log 2>&1 &

echo "GA LLM process started in background. PID: $!"
echo "Logs are being written to: /root/project/EvoLite/src/ga/ga_llm_math.log"
