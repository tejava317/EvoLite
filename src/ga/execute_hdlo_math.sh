#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evolite

cd /root/project/EvoLite
export PYTHONUNBUFFERED=1

nohup python -u -m src.ga.hdlo \
    --task MATH \
    --num-problems 20 \
    --max-rounds 20 \
    --patience 4 \
    --server-url http://localhost:8002 \
    --batch-size 15 \
    --proposals-per-round 7 \
    --max-front-size 15 \
    --run-id hdlo_math_v2 \
    > /root/project/EvoLite/src/ga/hdlo_math.log 2>&1 &

echo "HDLO process started in background. PID: $!"
echo "Logs are being written to: /root/project/EvoLite/src/ga/hdlo_math.log"

