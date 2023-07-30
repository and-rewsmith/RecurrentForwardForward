# Run benchmarks
python -m RecurrentFF.benchmarks.mnist.mnist --config-file ./test/config-files/smoke.toml
python -m RecurrentFF.benchmarks.moving_mnist.moving_mnist --config-file ./test/config-files/smoke.toml
python -m RecurrentFF.benchmarks.seq_mnist.seq --config-file ./test/config-files/smoke.toml
