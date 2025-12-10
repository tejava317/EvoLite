category="small-adaptive"

nohup python -m src.ga.ga \
    --population-size 30 \
    --generation 30 \
    --max-workflow 5 \
    --no-extractor \
    --key "$category" \
    > "src/ga/result/${category}.txt" 2>&1 &
