#!/bin/bash
set -e

export WANDB_MODE=dryrun
python -m RecurrentFF.benchmarks.mnist.mnist --config-file ./test/config-files/ci.toml > log.txt

# Extract the accuracy from the log
accuracy=$(grep "Test accuracy" log.txt | awk -F': ' '{print $NF}')
accuracy=${accuracy%\%}  # Remove the % sign from the accuracy
echo "Test Accuracy: $accuracy%"

# Compare the accuracy to the threshold
threshold=63
if (( $(echo "$accuracy >= $threshold" | bc -l) )); then
    echo "Test passed. Accuracy ($accuracy%) is above the threshold ($threshold%)."
else
    echo "Test failed. Accuracy ($accuracy%) is below the threshold ($threshold%)."
fi