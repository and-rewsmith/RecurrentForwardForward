import os
import multiprocessing

import torch
from torch.utils.data import DataLoader, Dataset
import wandb
from RecurrentFF.model.data_scenario.static_single_class import SingleStaticClassTestData

from RecurrentFF.settings import Settings
from RecurrentFF.util import DataConfig, TrainInputData, TrainLabelData, set_logging
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.benchmarks.moving_mnist.constants import MOVING_MNIST_DATA_DIR

NUM_CLASSES = 10
DATA_SIZE = 4096
TRAIN_BATCH_SIZE = 5000
TEST_BATCH_SIZE = 5000
ITERATIONS = 20
FOCUS_ITERATION_NEG_OFFSET = 2
FOCUS_ITERATION_POS_OFFSET = 2
DATA_PER_FILE = 1000


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

    def __init__(self, root_dir, train=True):
        """
        Initializes the dataset with the root directory, the training/testing mode, and the max size of the queue.
        It also initializes the data queue and loads the first chunk of data into memory.
        """
        self.root_dir = root_dir
        self.train = train

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

    def __len__(self):
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

    def __getitem__(self, idx):
        """
        Returns the sequence and its corresponding positive and negative labels at the given index.
        If the index is beyond the current data chunk, it loads the next chunk into memory.

        Parameters:
        -----------
        idx : int
            The index of the sequence to get.

        Returns:
        --------
        Tuple:
            A tuple containing two sequences and their corresponding positive and negative labels.
            Each sequence is a tensor and the labels are one-hot encoded tensors.
        """
        if idx >= self.data_chunk_idx + len(self.data_chunk["sequences"]):
            self.file_idx += 1

            self.data_chunk = torch.load(os.path.join(
                self.root_dir, self.data_files[self.file_idx]))

            # Update the chunk start index
            self.data_chunk_idx += len(self.data_chunk["sequences"])

        sequences = self.data_chunk['sequences'][idx -
                                                 self.data_chunk_idx][0:ITERATIONS]
        sequences = sequences.view(sequences.shape[0], -1)

        y_pos = self.data_chunk['labels'][idx - self.data_chunk_idx]
        y_neg = y_pos
        while y_neg == y_pos:
            y_neg = torch.randint(0, NUM_CLASSES, (1,)).item()

        positive_one_hot_labels = torch.zeros(NUM_CLASSES)
        positive_one_hot_labels[y_pos] = 1.0

        negative_one_hot_labels = torch.zeros(NUM_CLASSES)
        negative_one_hot_labels[y_neg] = 1.0

        return (sequences, sequences), (positive_one_hot_labels,
                                        negative_one_hot_labels)


def train_collate_fn(batch):
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


def test_collate_fn(batch):
    data_batch, label_batch = zip(*batch)

    pos_data, _neg_data = zip(*data_batch)
    positive_labels, _negative_labels = zip(*label_batch)

    pos_data = torch.stack(pos_data, 1)
    positive_labels = torch.stack(positive_labels)

    positive_labels = positive_labels.argmax(dim=1)

    return SingleStaticClassTestData(pos_data, positive_labels)


def MNIST_loaders(train_batch_size, test_batch_size):
    # TODO: need a transform? Similar to MNIST:
    # transform = Compose([
    #     ToTensor(),
    #     Normalize((0.1307,), (0.3081,)),
    #     Lambda(lambda x: torch.flatten(x))])

    # Cannot shuffle with the dataset implementation
    train_loader = DataLoader(
        MovingMNISTDataset(
            f'{MOVING_MNIST_DATA_DIR}/',
            train=True),
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=train_collate_fn,
        num_workers=0)

    test_loader = DataLoader(
        MovingMNISTDataset(
            f'{MOVING_MNIST_DATA_DIR}/',
            train=False),
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
        "focus_iteration_neg_offset": FOCUS_ITERATION_NEG_OFFSET,
        "focus_iteration_pos_offset": FOCUS_ITERATION_POS_OFFSET}

    if settings.data_config is None:
        settings.data_config = DataConfig(**data_config)

    set_logging()

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",

        # track hyperparameters and run metadata
        config={
            "architecture": "Recurrent-FF",
            "dataset": "Moving-MNIST",
            "settings": settings.model_dump(),
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)

    model.train(train_loader, test_loader)

    # Explicitly delete multiprocessing components
    del train_loader
    del test_loader
