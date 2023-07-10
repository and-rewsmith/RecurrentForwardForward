# Recurrent Forward Forward Network

The implementation of the Recurrent Forward Forward Network is based on the [following paper](https://arxiv.org/abs/2212.13345). A three layer implementation of this network is benchmarked on MNIST achieving a ~94% test accuracy with < 1k neurons.

![Recurrent Forward Forward Network](img/Fig3.png "Recurrent Forward Forward")

## Usage

```
pip install -e .
python -m RecurrentFF.benchmarks.MNIST.mnist
```
