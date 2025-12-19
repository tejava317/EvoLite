#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evolite

cd /root/project/EvoLite
export PYTHONUNBUFFERED=1

nohup python -u -m src.ga.ga \
    --task MBPP \
    --population-size 30 \
    --generation 20 \
    --num_problem 20 \
    --batch-size 30 \
    --server-url http://localhost:8002 \
    --key mbpp_ga_v1 \
    > /root/project/EvoLite/src/ga/ga_mbpp.log 2>&1 &

echo "GA process started in background. PID: $!"
echo "Logs are being written to: /root/project/EvoLite/src/ga/ga_mbpp.log"
