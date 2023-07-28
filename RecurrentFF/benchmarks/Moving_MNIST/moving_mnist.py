import os
import logging
import time

import torch
from torch.utils.data import DataLoader, Dataset
import wandb
from matplotlib import pyplot as plt
from multiprocessing import Process, Manager

from RecurrentFF.model.model import RecurrentFFNet, TrainInputData, TrainLabelData, SingleStaticClassTestData
from RecurrentFF.model.constants import EPOCHS, LEARNING_RATE, THRESHOLD, DAMPING_FACTOR, EPSILON, DEVICE

from RecurrentFF.benchmarks.Moving_MNIST.constants import MOVING_MNIST_DATA_DIR

NUM_CLASSES = 10
INPUT_SIZE = 4096
LAYERS = [1000, 1000, 1000]
TRAIN_BATCH_SIZE = 5000
TEST_BATCH_SIZE = 5000
ITERATIONS = 10
DATA_PER_FILE = 1000


class MovingMNISTDataset(Dataset):
    """
    Custom Dataset class for loading and processing the Moving MNIST dataset.

    The data are loaded asynchronously in a separate process to avoid I/O
    blocking. The class is designed to work with .pt files and supports
    train/test splitting based on filename prefix.

    NOTE: Do not use shuffle=True for the dataloader. It is not supported.

    Parameters
    ----------
    root_dir : str
        Path to the directory with .pt files.
    train : bool, optional
        If True, the dataset will only load files with 'train_' prefix. If
        False, only 'test_' files are loaded.
    queue_maxsize : int, optional
        The maximum size of the data loading queue.

    Attributes
    ----------
    root_dir : str
        Path to the directory with .pt files.
    train : bool
        If True, the dataset will only load files with 'train_' prefix.
    queue_maxsize : int
        The maximum size of the data loading queue.
    data_files : list
        List of .pt files to be loaded.
    file_idx : int
        Index of the current file being processed.
    data_chunk_idx : int
        Index of the current chunk in memory.
    data_queue : multiprocessing.Manager().Queue()
        A queue object used for loading data chunks.
    load_event : multiprocessing.Manager().Event()
        An event object used for signaling when new data should be loaded.
    loader_process : multiprocessing.Process()
        A separate process for loading data chunks.
    data_chunk : dict
        The current data chunk in memory.
    """

    def __init__(self, root_dir, train=True, queue_maxsize=5):
        self.root_dir = root_dir
        self.train = train
        self.queue_maxsize = queue_maxsize

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

        manager = Manager()
        self.data_queue = manager.Queue(
            maxsize=self.queue_maxsize)  # Parameterized queue size
        self.load_event = manager.Event()

        chunk_data = torch.load(os.path.join(
            self.root_dir, self.data_files[self.file_idx]))
        self.data_queue.put(chunk_data)
        self.file_idx += 1

        self.loader_process = Process(
            target=self._load_data_loop, args=(self.data_queue, self.load_event))
        self.loader_process.start()

        self.data_chunk = self.data_queue.get()  # Load first chunk into memory

    def __len__(self):
        # this will be called at beginning of new batch, so reset is needed
        self.data_chunk_idx = 0
        return len(self.data_files) * DATA_PER_FILE

    def __getitem__(self, idx):
        """
        Fetches a data instance and its corresponding label from the dataset.

        If the requested index goes beyond the current data chunk, the method
        signals for the next chunk to be loaded.

        Parameters
        ----------
        idx : int
            Index of the data instance to fetch.

        Returns
        -------
        tuple
            Tuple of sequences and corresponding one-hot labels.
        """
        while idx >= self.data_chunk_idx + len(self.data_chunk["sequences"]):
            self.load_event.set()  # Signal the background process to load more data

            while self.data_queue.empty():
                time.sleep(0.1)  # Wait for data to be loaded
                pass

            self.data_chunk = self.data_queue.get()  # Wait for data to be loaded
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

        return (sequences, sequences), (positive_one_hot_labels, negative_one_hot_labels)

    def _load_data_loop(self, data_queue, load_event):
        """
        An internal method that runs in a separate process and loads data
        chunks.

        The method waits for a load event before loading data. It then loads as
        many chunks as it can into the queue.

        Parameters
        ----------
        data_queue : multiprocessing.Manager().Queue()
            The queue to which data chunks are loaded.
        load_event : multiprocessing.Manager().Event()
            An event object used for signaling when new data should be loaded.
        """
        while True:
            load_event.wait()  # Wait for a signal to load data

            # Load chunks into the queue until it is full or all data files have been loaded
            while not data_queue.full() and self.file_idx < len(self.data_files):
                chunk_data = torch.load(os.path.join(
                    self.root_dir, self.data_files[self.file_idx]))
                data_queue.put(chunk_data)
                self.file_idx += 1

            load_event.clear()  # Clear the event after data is loaded

            # reset the loop
            if self.file_idx >= len(self.data_files):
                self.file_idx = 0


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

    return TrainInputData(data1, data2), TrainLabelData(positive_labels, negative_labels)


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
    train_loader = DataLoader(MovingMNISTDataset(f'{MOVING_MNIST_DATA_DIR}/', train=True),
                              batch_size=train_batch_size, shuffle=False, collate_fn=train_collate_fn, num_workers=0)

    test_loader = DataLoader(MovingMNISTDataset(f'{MOVING_MNIST_DATA_DIR}/', train=False),
                             batch_size=test_batch_size, shuffle=False, collate_fn=test_collate_fn, num_workers=0)

    return train_loader, test_loader


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    print("init wandb")
    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",

        # track hyperparameters and run metadata
        config={
            "architecture": "Recurrent-FF",
            "dataset": "Moving-MNIST",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "layers": str(LAYERS),
            "iterations": ITERATIONS,
            "threshold": THRESHOLD,
            "damping_factor": DAMPING_FACTOR,
            "epsilon": EPSILON,
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    # Create and run model.
    model = RecurrentFFNet(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
                           INPUT_SIZE, LAYERS, NUM_CLASSES).to(DEVICE)

    model.train(train_loader, test_loader)

    model.predict(test_loader)
