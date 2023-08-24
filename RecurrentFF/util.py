from enum import Enum
import logging
import torch
from torch import nn

from RecurrentFF.settings import Settings


# NOTE: Square images only
def calculate_conv_output_dimensions(data_size,
                                     output_channels,
                                     kernel_size,
                                     conv_stride,
                                     conv_padding,
                                     max_pool_kernel_size,
                                     max_pool_stride):

    square_image_width = int(data_size**0.5)
    H_in = square_image_width
    W_in = square_image_width

    # Output dimensions after the convolutional layer
    H_out = ((H_in - kernel_size + 2 * conv_padding) // conv_stride) + 1
    W_out = ((W_in - kernel_size + 2 * conv_padding) // conv_stride) + 1

    # Output dimensions after the max pooling layer
    H_out_pool = ((H_out - max_pool_kernel_size) // max_pool_stride) + 1
    W_out_pool = ((W_out - max_pool_kernel_size) // max_pool_stride) + 1

    # Multiply spatial dimensions by output channels
    return output_channels * H_out_pool * W_out_pool


def set_logging():
    """
    Must be called after argparse.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def standardize_layer_activations(layer_activations, epsilon):

    # Compute mean and standard deviation for prev_layer
    prev_layer_mean = layer_activations.mean(
        dim=1, keepdim=True)
    prev_layer_std = layer_activations.std(
        dim=1, keepdim=True)

    # Apply standardization
    prev_layer_stdized = (layer_activations - prev_layer_mean) / \
        (prev_layer_std + epsilon)

    return prev_layer_stdized


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


def layer_activations_to_badness(layer_activations):
    """
    Computes the 'badness' of activations for a given layer in a neural network
    by taking the mean of the squared values.

    'Badness' in this context refers to the average squared activation value.
    This function is designed to work with PyTorch tensors, which represent the
    layer's activations.

    Args:
        layer_activations (torch.Tensor): A tensor representing activations from
        one layer of a neural network. The tensor has shape (batch_size,
        num_activations), where batch_size is the number of samples processed
        together, and num_activations is the number of neurons in the layer.

    Returns:
        torch.Tensor: A tensor corresponding to the 'badness' (mean of the
        squared activations) of the given layer. The output tensor has shape
        (batch_size,), since the mean is taken over the activation values for
        each sample in the batch.
    """
    badness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1)

    return badness_for_layer


class LatentAverager:
    """
    This class is used for tracking and averaging tensors of the same shape.
    It's useful for collapsing latents in a series of computations.
    """

    def __init__(self):
        """
        Initialize the LatentAverager with an empty sum_tensor and a zero count.
        """
        self.sum_tensor = None
        self.count = 0

    def track_collapsed_latents(self, tensor: torch.Tensor):
        """
        Add the given tensor to the tracked sum.

        :param tensor: A tensor to be tracked.
        :type tensor: torch.Tensor
        :raises AssertionError: If the shape of the tensor does not match the shape of the sum_tensor.
        """
        if self.sum_tensor is None:
            self.sum_tensor = tensor
            self.count = 1
        else:
            assert tensor.shape == self.sum_tensor.shape, "Shape mismatch"
            self.sum_tensor += tensor
            self.count += 1

    def retrieve(self) -> torch.Tensor:
        """
        Retrieve the averaged tensor.

        :return: The averaged tensor.
        :rtype: torch.Tensor
        :raises ValueError: If no tensors have been tracked.
        """
        if self.count == 0:
            raise ValueError("No tensors have been tracked")
        return self.sum_tensor / self.count
