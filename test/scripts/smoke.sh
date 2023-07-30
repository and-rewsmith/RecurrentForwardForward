# Run benchmarks not needing data files
export WANDB_MODE=dryrun
python -m RecurrentFF.benchmarks.mnist.mnist --config-file ./test/config-files/smoke.toml
