import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import wandb
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from RecurrentFF.model.data_scenario.static_single_class import SingleStaticClassTestData
from RecurrentFF.settings import Settings, DataConfig
from RecurrentFF.util import TrainInputData, TrainLabelData, set_logging
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.benchmarks.moving_mnist.constants import MOVING_MNIST_DATA_DIR

DATA_SIZE = 4096
NUM_CLASSES = 10
TRAIN_BATCH_SIZE = 500
TEST_BATCH_SIZE = 500
ITERATIONS = 50
DATASET = "Moving-MNIST"

DATA_PER_FILE = 1000
PATCH_SIZE = 8


class MovingMNISTDataset(Dataset):
    """
    A Dataset for loading and serving sequences from the MovingMNIST dataset. This dataset is specifically
    designed for loading .pt (PyTorch serialized tensors) files from a directory and organizing them into
    chunks that can be fed to a model for training or testing. The dataset uses multiprocessing queues
    for loading the chunks, enabling concurrent data loading and processing. It also separates training
    and testing data based on the filenames, allowing for easy dataset splitting.

    Attributes:
    -----------
    root_dir : str
        The directory where the .pt files are stored.
    train : bool
        Specifies if the dataset should load the training files or the testing files.
    data_files : list
        The list of data file paths loaded from the root_dir.
    file_idx : int
        The index of the current file being processed.
    data_chunk_idx : int
        The index of the current chunk in memory.
    load_event : multiprocessing.Queue
        A queue to signal data loading events.
    data_chunk : dict
        The current chunk of data in memory. The chunk is a dictionary with "sequences" and "labels" as keys.
    """

    def __init__(self, root_dir, train=True, patch_size=PATCH_SIZE) -> None:
        """
        Initializes the dataset with the root directory, the training/testing mode, and the max size of the queue.
        It also initializes the data queue and loads the first chunk of data into memory.
        """
        self.root_dir = root_dir
        self.train = train

        self.patch_size = patch_size

        # List of all .pt files in root_dir
        self.data_files = [f for f in os.listdir(
            self.root_dir) if f.endswith('.pt')]

        # Separate train and test files
        if self.train:
            self.data_files = [
                f for f in self.data_files if f.startswith('train_')]
        else:
            self.data_files = [
                f for f in self.data_files if f.startswith('test_')]

        self.file_idx = 0  # Index of the current file
        self.data_chunk_idx = 0  # Index of the current chunk in memory

        self.data_chunk = torch.load(os.path.join(
            self.root_dir, self.data_files[self.file_idx]))

    def __len__(self) -> int:
        """
        Returns the total number of sequences in the dataset. This method also resets the counters,
        making ready for the next epoch.

        Returns:
        --------
        int:
            The total number of sequences in the dataset.
        """
        # this will be called at beginning of new epoch, so reset is needed
        self.data_chunk_idx = 0
        self.file_idx = 0
        return len(self.data_files) * DATA_PER_FILE

    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if idx >= self.data_chunk_idx + len(self.data_chunk["sequences"]):
            self.file_idx += 1
            self.data_chunk = torch.load(os.path.join(
                self.root_dir, self.data_files[self.file_idx]))
            self.data_chunk_idx += len(self.data_chunk["sequences"])

        # Get the sequence and reshape it
        sequences = self.data_chunk['sequences'][idx -
                                                 self.data_chunk_idx][0:ITERATIONS]

        # Transform each frame in the sequence to patches
        patched_sequences = []
        for frame in sequences:
            # Each frame should be 28x28
            patches = mnist_to_patches(frame, self.patch_size)
            patched_sequences.append(patches)

        # Stack the patched sequences
        patched_sequences = torch.stack(patched_sequences)

        # Handle labels
        y_pos = self.data_chunk['labels'][idx - self.data_chunk_idx]
        y_neg = y_pos
        while y_neg == y_pos:
            y_neg = torch.randint(0, NUM_CLASSES, (1,)).item()

        positive_one_hot_labels = torch.zeros(NUM_CLASSES)
        positive_one_hot_labels[y_pos] = 1.0

        negative_one_hot_labels = torch.zeros(NUM_CLASSES)
        negative_one_hot_labels[y_neg] = 1.0

        # print(patched_sequences.shape, positive_one_hot_labels.shape,
        #       negative_one_hot_labels.shape)
        # input()
        return (patched_sequences, patched_sequences), (positive_one_hot_labels, negative_one_hot_labels)


def mnist_to_patches(img, patch_size):
    """
    Convert an image into patches.

    Args:
        img (torch.Tensor): Input image tensor of shape [H, W] or [1, H, W]
        patch_size (int): Size of each square patch

    Returns:
        torch.Tensor: Flattened patches tensor
    """
    # Handle both single channel and no channel inputs
    if len(img.shape) == 2:
        # Add channel dimension if not present
        img = img.unsqueeze(0)

    # Get dimensions
    C, H, W = img.shape
    assert C == 1, "Image should be single channel"
    assert H % patch_size == 0 and W % patch_size == 0, f"Image dimensions ({H}, {W}) must be divisible by patch size {patch_size}"

    # Calculate number of patches along each dimension
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Rearrange the image into patches
    patches = img.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size)
    # Now patches has shape [1, num_patches_h, num_patches_w, patch_size, patch_size]

    # Remove channel dim and reshape to [num_patches_total, patch_size*patch_size]
    patches = patches.squeeze(0)
    patches = patches.reshape(-1, patch_size * patch_size)

    return patches.flatten()


def train_collate_fn(batch) -> TrainInputData:
    data_batch, label_batch = zip(*batch)

    data1, data2 = zip(*data_batch)
    positive_labels, negative_labels = zip(*label_batch)

    data1 = torch.stack(data1, 1)
    data2 = torch.stack(data2, 1)
    positive_labels = torch.stack(positive_labels)
    negative_labels = torch.stack(negative_labels)

    positive_labels = positive_labels.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    negative_labels = negative_labels.unsqueeze(0).repeat(ITERATIONS, 1, 1)

    return TrainInputData(
        data1, data2), TrainLabelData(
        positive_labels, negative_labels)


def test_collate_fn(batch) -> SingleStaticClassTestData:
    data_batch, label_batch = zip(*batch)

    pos_data, _neg_data = zip(*data_batch)
    positive_labels, _negative_labels = zip(*label_batch)

    pos_data = torch.stack(pos_data, 1)
    positive_labels = torch.stack(positive_labels)

    positive_labels = positive_labels.argmax(
        dim=1).unsqueeze(0).repeat(ITERATIONS, 1)

    return SingleStaticClassTestData(pos_data, positive_labels)


def MNIST_loaders(train_batch_size, test_batch_size) -> Tuple[DataLoader, DataLoader]:
    # TODO: need a transform? Similar to MNIST:
    # transform = Compose([
    #     ToTensor(),
    #     Normalize((0.1307,), (0.3081,)),
    #     Lambda(lambda x: torch.flatten(x))])

    transform = Compose([
        Lambda(lambda x: mnist_to_patches(x, 4)),
    ])

    # Cannot shuffle with the dataset implementation
    train_loader = DataLoader(
        MovingMNISTDataset(
            f'{MOVING_MNIST_DATA_DIR}/',
            train=True,
            patch_size=PATCH_SIZE),
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=train_collate_fn,
        num_workers=0)

    test_loader = DataLoader(
        MovingMNISTDataset(
            f'{MOVING_MNIST_DATA_DIR}/',
            train=False,
            patch_size=PATCH_SIZE),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=0)

    return train_loader, test_loader


if __name__ == '__main__':
    settings = Settings.new()

    data_config = {
        "data_size": DATA_SIZE,
        "num_classes": NUM_CLASSES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "iterations": ITERATIONS,
        "dataset": DATASET,
    }

    if settings.data_config is None:
        settings.data_config = DataConfig(**data_config)

    set_logging()

    # Pytorch utils.
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",

        # track hyperparameters and run metadata
        config={
            "architecture": "Recurrent-FF",
            "dataset": DATASET,
            "settings": settings.model_dump(),
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)

    model.train_model(train_loader, test_loader)

    # Explicitly delete multiprocessing components
    del train_loader
    del test_loader
