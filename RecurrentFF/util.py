from enum import Enum
import logging
from typing import Generator

import torch
from torch.nn import functional as F


def shuffle_softmax(tensor):
    """
    Shuffles the values in the softmax dimension of the input tensor randomly.

    Args:
    tensor (torch.Tensor): A 2D tensor of shape (batch_size, softmax_dimension)

    Returns:
    torch.Tensor: A tensor with the same shape where values within each batch are shuffled randomly
    """
    # Get the batch size and softmax dimension from the tensor
    batch_size, softmax_dim = tensor.shape

    # Initialize an empty tensor to hold the shuffled values
    shuffled_tensor = torch.empty_like(tensor)

    # Iterate over each batch
    for i in range(batch_size):
        # Shuffle the values in the softmax dimension for this batch
        shuffled_tensor[i] = tensor[i, torch.randperm(softmax_dim)]

    return shuffled_tensor


def sample_from_logits_excluding_highest(logits):
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    # Find the index of the highest probability class
    _, max_prob_indices = torch.max(probs, dim=1)

    # Create a mask to exclude the highest probability class
    mask = torch.ones_like(probs).scatter_(1, max_prob_indices.unsqueeze(1), 0)

    # Apply the mask to the probabilities
    masked_probs = probs * mask

    # Normalize the masked probabilities
    normalized_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)

    # Handle the case where all probabilities were equal (resulting in all zeros after masking)
    zero_rows = (normalized_probs.sum(dim=1) == 0)
    if zero_rows.any():
        # For these rows, we'll sample uniformly from all classes except the max
        uniform_probs = mask.float() / (mask.sum(dim=1, keepdim=True) - 1)
        normalized_probs[zero_rows] = uniform_probs[zero_rows]

    # Replace NaNs with zero and set any Infs or negative values to zero
    safe_probs = torch.nan_to_num(
        normalized_probs, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure all values are non-negative
    safe_probs = torch.clamp(safe_probs, min=0.0)

    # Re-normalize the probabilities to ensure they sum to 1
    if safe_probs.sum() > 0:
        safe_probs /= safe_probs.sum()
    else:
        # If all values are zero, create a uniform distribution as a fallback
        safe_probs = torch.ones_like(safe_probs) / safe_probs.numel()

    # Sample from the modified and safe probability distribution
    try:
        sampled_indices = torch.multinomial(
            safe_probs, num_samples=1).squeeze()
        # Create a one-hot vector from the sampled indices
        one_hot = F.one_hot(
            sampled_indices, num_classes=logits.size(1)).float()
    except RuntimeError as e:
        print(f"-----error! {str(e)}")
        one_hot = torch.zeros_like(safe_probs)

    return one_hot


def sample_avoiding_correct_class(logits, correct_classes):
    # Clone to avoid modifying original logits in-place
    modified_logits = logits.clone()

    # Set a large negative value for logits at the index of the correct class
    modified_logits[correct_classes == 1] = -1e9

    # Apply softmax to the modified logits
    probabilities = F.softmax(modified_logits, dim=1)

    # Sample from the probability distribution
    sampled_indices = torch.multinomial(probabilities, num_samples=1).squeeze()

    # Create a one-hot vector from the sampled indices
    one_hot = F.one_hot(sampled_indices, num_classes=logits.size(1)).float()

    return one_hot


def zero_correct_class_softmax(logits, correct_classes):
    # Set a large negative value for logits at the index of the correct class
    # Clone to avoid modifying original logits in-place
    modified_logits = logits.clone()
    modified_logits[correct_classes == 1] = - \
        1e9  # Use a very large negative value

    # Apply softmax to the modified logits
    softmax_output = F.softmax(modified_logits, dim=1)
    return softmax_output


def zero_highest_logit(logits):
    # Clone logits to avoid in-place modification
    modified_logits = logits.clone()

    # Find the index of the highest logit value for each row (batch)
    highest_logit_indices = torch.argmax(modified_logits, dim=1, keepdim=True)

    # Set the highest logit values to a large negative number (e.g., -1e9) to zero them out after softmax
    modified_logits.scatter_(1, highest_logit_indices, -1e9)

    # Apply softmax to the modified logits
    softmax_output = F.softmax(modified_logits, dim=1)
    return softmax_output

# Function to calculate the percentage of cases where the argmax matches the correct label


def percent_correct(softmax_output, correct_labels):
    # Convert one-hot encoded labels to indices (get the correct class indices)
    correct_indices = torch.argmax(correct_labels, dim=1)  # [batch_size]

    # Get the predicted class from softmax output (argmax of predictions)
    predicted_indices = torch.argmax(softmax_output, dim=1)  # [batch_size]

    # Compare predicted indices with the correct indices
    num_correct = torch.sum(predicted_indices == correct_indices).item()

    # Calculate the percentage of correct predictions
    percent_correct = (num_correct / correct_indices.size(0)) * 100

    # Return the percentage of correct argmax predictions
    return percent_correct

# Function to check the percentage of correct softmax values above a given threshold


def percent_above_threshold(softmax_output, correct_labels, confidence_threshold):
    # Convert one-hot encoded labels to indices (get the correct class indices)
    correct_indices = torch.argmax(correct_labels, dim=1)  # [batch_size]

    # Gather the softmax probabilities of the correct class for each example in the batch
    correct_class_probs = softmax_output.gather(
        1, correct_indices.unsqueeze(1)).squeeze()

    # Check how many of the correct class probabilities are above the confidence threshold
    num_above_threshold = torch.sum(
        correct_class_probs > confidence_threshold).item()

    # Calculate the percentage of softmax values that are above the threshold
    percent_above = (num_above_threshold / correct_class_probs.size(0)) * 100

    # Return the percentage
    return percent_above


# Updated function without prints and assuming label data is passed directly
def is_confident(softmax_output, correct_labels, confidence_threshold):
    # Convert one-hot encoded labels to indices (get the correct class indices)
    correct_indices = torch.argmax(correct_labels, dim=1)  # [batch_size]

    # Gather the softmax probabilities of the correct class for each example in the batch
    correct_class_probs = softmax_output.gather(
        1, correct_indices.unsqueeze(1)).squeeze()

    # Check if all of the correct class probabilities are above the confidence threshold
    all_confident = torch.all(correct_class_probs > confidence_threshold)

    # Calculate the average confidence
    avg_confidence = torch.mean(correct_class_probs)

    # Return the confidence probabilities, average confidence, and whether all are above the threshold
    return avg_confidence.item(), all_confident.item()


def swap_top_two_softmax(tensor):
    # Find the top two values along the softmax dimension
    top2_values, top2_indices = torch.topk(tensor, 2, dim=1)

    # Clone the original tensor so we can modify it
    swapped_tensor = tensor.clone()

    # Create a tensor of batch indices
    batch_indices = torch.arange(tensor.size(0)).unsqueeze(1)

    # Get the indices of the top two values
    max_indices = top2_indices[:, 0]  # Highest value indices
    second_max_indices = top2_indices[:, 1]  # Second highest value indices

    # Swap the values
    swapped_tensor[batch_indices, max_indices] = top2_values[:, 1]
    swapped_tensor[batch_indices, second_max_indices] = top2_values[:, 0]

    return swapped_tensor


def set_logging() -> None:
    """
    Must be called after argparse.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def standardize_layer_activations(layer_activations: torch.Tensor, epsilon: float = 0.000000000001) -> torch.Tensor:
    # squared_activations = layer_activations ** 2
    # mean_squared = torch.mean(squared_activations, dim=1, keepdim=True)
    # l2_norm = torch.sqrt(mean_squared + epsilon)

    # normalized_activations = layer_activations / l2_norm
    # return normalized_activations
    return layer_activations


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
