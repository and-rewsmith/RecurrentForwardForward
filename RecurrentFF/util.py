from enum import Enum
import logging
from typing import Generator

from torch import Tensor
import torch


def set_logging() -> None:
    """
    Must be called after argparse.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def standardize_layer_activations(layer_activations: torch.Tensor, epsilon: float) -> torch.Tensor:
    squared_activations = layer_activations ** 2
    mean_squared = torch.mean(squared_activations, dim=1, keepdim=True)
    l2_norm = torch.sqrt(mean_squared + epsilon)

    normalized_activations = layer_activations / l2_norm
    return normalized_activations


def scale_labels_by_timestep_train(label_data: tuple[Tensor, Tensor], iteration, max_iterations) -> tuple[Tensor, Tensor]:
    # tensor of shape (batch_size, classes)
    pos_labels = scale_labels_by_timestep(
        label_data[0], iteration, max_iterations)
    neg_labels = scale_labels_by_timestep(
        label_data[1], iteration, max_iterations)
    return (pos_labels, neg_labels)


def scale_labels_by_timestep(label_data: Tensor, iteration: int, max_iterations: int) -> Tensor:
    # tensor of shape (batch_size, classes)
    return label_data * (iteration / max_iterations)


class TrainTestBridgeFormatLoader:
    def __init__(self, train_loader: torch.utils.data.DataLoader) -> None:
        self.train_loader = train_loader

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        for input_data, label_data in self.train_loader:
            pos_input = input_data.pos_input
            pos_labels = label_data.pos_labels
            pos_labels = pos_labels.argmax(dim=2)
            yield pos_input, pos_labels


class TrainInputData:
    """
    input of dims (frames, batch size, input size)
    """

    def __init__(self, pos_input: torch.Tensor, neg_input: torch.Tensor) -> None:
        self.pos_input = pos_input
        self.neg_input = neg_input

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        yield self.pos_input
        yield self.neg_input

    def move_to_device_inplace(self, device: str) -> None:
        self.pos_input = self.pos_input.to(device)
        self.neg_input = self.neg_input.to(device)


# input of dims (frames, batch size, num classes)
class TrainLabelData:
    def __init__(self, pos_labels: torch.Tensor, neg_labels: torch.Tensor) -> None:
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        yield self.pos_labels
        yield self.neg_labels

    def move_to_device_inplace(self, device: str) -> None:
        self.pos_labels = self.pos_labels.to(device)
        self.neg_labels = self.neg_labels.to(device)


class Activations:
    def __init__(self, current: torch.Tensor, previous: torch.Tensor) -> None:
        self.current = current
        self.previous = previous

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        yield self.current
        yield self.previous

    def advance(self) -> None:
        self.previous = self.current


class ForwardMode(Enum):
    PositiveData = 1
    NegativeData = 2
    PredictData = 3


def layer_activations_to_badness(layer_activations: torch.Tensor) -> torch.Tensor:
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

    def __init__(self) -> None:
        """
        Initialize the LatentAverager with an empty sum_tensor and a zero count.
        """
        self.sum_tensor: torch.Tensor = None  # type: ignore[assignment]
        self.count = 0

    def track_collapsed_latents(self, tensor: torch.Tensor) -> None:
        """
        Add the given tensor to the tracked sum.
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
        """
        if self.count == 0:
            raise ValueError("No tensors have been tracked")
        return self.sum_tensor / self.count
