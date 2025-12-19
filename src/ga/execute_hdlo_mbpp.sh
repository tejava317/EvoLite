#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evolite

cd /root/project/EvoLite
export PYTHONUNBUFFERED=1

python -u -m src.ga.hdlo \
    --task MBPP \
    --num-problems 20 \
    --max-rounds 20 \
    --patience 4 \
    --server-url http://localhost:8002 \
    --batch-size 15 \
    --proposals-per-round 5 \
    --run-id hdlo_mbpp_v1

