#!/bin/bash
set -e

export WANDB_MODE=dryrun

# Capture the output of the script in a variable
output=$(python -m RecurrentFF.benchmarks.mnist.mnist --config-file ./test/config-files/ci.toml)

# Extract the accuracy from the output
accuracy=$(echo "$output" | grep "test accuracy" | awk -F': ' '{print $NF}')
accuracy=${accuracy%\%}  # Remove the % sign from the accuracy
echo "Test Accuracy: $accuracy%"

# Compare the accuracy to the threshold
threshold=87.7
if (( $(echo "$accuracy >= $threshold" | bc -l) )); then
    echo "Test passed. Accuracy ($accuracy%) is above the threshold ($threshold%)."
else
    echo "Test failed. Accuracy ($accuracy%) is below the threshold ($threshold%)."
    exit 1
fi
