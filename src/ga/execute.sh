category="single-ga"

nohup python -m src.ga.ga \
    --task MBPP \
    --population-size 50 \
    --generation 30 \
    --max-workflow 5 \
    --no-extractor \
    --num_phase 1 \
    --key "$category" \
    > "src/ga/result/${category}.txt" 2>&1 &
