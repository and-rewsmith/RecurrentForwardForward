# Recurrent Forward Forward Network

The implementation of the Recurrent Forward Forward Network is based on the [following paper](https://arxiv.org/abs/2212.13345). A three layer implementation of this network is benchmarked on MNIST achieving a ~94% test accuracy with < 1k neurons.

![Recurrent Forward Forward Network](img/Fig3.png "Recurrent Forward Forward")

## Usage

```
pip install -e .
python -m RecurrentFF.benchmarks.MNIST.mnist
```

## Model TODO:

- [x] Recurrent connections
- [x] Lateral connections
- [x] Data and label inputs conducive to changing accross timesteps
- [ ] Dynamic negative data
- [ ] Invert objective function: low activations for positive data
- [ ] Receptive fields
- [ ] Peer normalization
- [ ] Support data manipulation for positive data
- [ ] Generative circuit
- [ ] Support data reconstruction
- [ ] Support negative data synthesis
- [ ] Making a spiking version of the model

## Benchmark TODO:

- [x] Benchmark on MNIST
- [x] Benchmark on Moving MNIST
- [x] Benchmark on Seq MNIST
- [ ] Benchmark on non-static multiclass dataset
