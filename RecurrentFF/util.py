from enum import Enum
import torch
from torch import nn

from RecurrentFF.settings import Settings


def standardize_layer_activations(layer_activations):
    settings = Settings()

    # Compute mean and standard deviation for prev_layer
    prev_layer_mean = layer_activations.mean(
        dim=1, keepdim=True)
    prev_layer_std = layer_activations.std(
        dim=1, keepdim=True)

    # Apply standardization
    prev_layer_stdized = (
        layer_activations - prev_layer_mean) / (prev_layer_std + settings.model.epsilon)

    return prev_layer_stdized


class DataConfig:
    def __init__(self, data_size, num_classes, train_batch_size, test_batch_size, iterations, focus_iteration_neg_offset, focus_iteration_pos_offset):
        self.data_size = data_size
        self.num_classes = num_classes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.iterations = iterations
        self.focus_iteration_neg_offset = focus_iteration_neg_offset
        self.focus_iteration_pos_offset = focus_iteration_pos_offset


# input of dims (frames, batch size, input size)
class TrainInputData:
    def __init__(self, pos_input, neg_input):
        self.pos_input = pos_input
        self.neg_input = neg_input

    def __iter__(self):
        yield self.pos_input
        yield self.neg_input

    def move_to_device_inplace(self, device):
        self.pos_input = self.pos_input.to(device)
        self.neg_input = self.neg_input.to(device)


# input of dims (frames, batch size, num classes)
class TrainLabelData:
    def __init__(self, pos_labels, neg_labels):
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels

    def __iter__(self):
        yield self.pos_labels
        yield self.neg_labels

    def move_to_device_inplace(self, device):
        self.pos_labels = self.pos_labels.to(device)
        self.neg_labels = self.neg_labels.to(device)


# input of dims (batch size, num classes)
class SingleStaticClassTestData:
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    def __iter__(self):
        yield self.input
        yield self.labels


class Activations:
    def __init__(self, current, previous):
        self.current = current
        self.previous = previous

    def __iter__(self):
        yield self.current
        yield self.previous

    def advance(self):
        self.previous = self.current


class OutputLayer(nn.Module):
    def __init__(self, prev_size, label_size) -> None:
        super(OutputLayer, self).__init__()

        self.backward_linear = nn.Linear(
            label_size, prev_size)


class ForwardMode(Enum):
    PositiveData = 1
    NegativeData = 2
    PredictData = 3


def activations_to_goodness(activations):
    """
    Computes the 'goodness' of activations for each layer in a neural network by
    taking the mean of the squared values.

    'Goodness' in this context refers to the average squared activation value.
    This function is designed to work with PyTorch tensors, which represent
    layers in a neural network.

    Args:
        activations (list of torch.Tensor): A list of tensors representing
        activations from each layer of a neural network. Each tensor in the list
        corresponds to one layer's activations, and has shape (batch_size,
        num_activations), where batch_size is the number of samples processed
        together, and num_activations is the number of neurons in the layer.

    Returns:
        list of torch.Tensor: A list of tensors, each tensor corresponding to
        the 'goodness' (mean of the squared activations) of each layer in the
        input. Each tensor in the output list has shape (batch_size,), since the
        mean is taken over the activation values for each sample in the batch.
    """
    goodness = []
    for act in activations:
        goodness_for_layer = torch.mean(
            torch.square(act), dim=1)
        goodness.append(goodness_for_layer)

    return goodness


def layer_activations_to_goodness(layer_activations):
    """
    Computes the 'goodness' of activations for a given layer in a neural network
    by taking the mean of the squared values.

    'Goodness' in this context refers to the average squared activation value.
    This function is designed to work with PyTorch tensors, which represent the
    layer's activations.

    Args:
        layer_activations (torch.Tensor): A tensor representing activations from
        one layer of a neural network. The tensor has shape (batch_size,
        num_activations), where batch_size is the number of samples processed
        together, and num_activations is the number of neurons in the layer.

    Returns:
        torch.Tensor: A tensor corresponding to the 'goodness' (mean of the
        squared activations) of the given layer. The output tensor has shape
        (batch_size,), since the mean is taken over the activation values for
        each sample in the batch.
    """
    goodness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1)

    return goodness_for_layer
