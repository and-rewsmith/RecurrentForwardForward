import logging

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import wandb

from RecurrentFF.model.model import RecurrentFFNet, TrainInputData, TrainLabelData, TestData
from RecurrentFF.model.constants import EPOCHS, LEARNING_RATE, ITERATIONS, THRESHOLD, DAMPING_FACTOR, EPSILON, DEVICE, ITERATIONS

NUM_CLASSES = 10
INPUT_SIZE = 784
LAYERS = [200, 200, 200]
TRAIN_BATCH_SIZE = 5000
TEST_BATCH_SIZE = 5000


class CustomTrainDataset(Dataset):
    """
    A custom PyTorch Dataset for training data that wraps around another dataset 
    and applies custom processing for positive and negative label generation.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to wrap around.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset and applies custom processing.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple ((pos_data, neg_data), (positive_one_hot_labels, negative_one_hot_labels)),
            where x is the sample data, and positive_one_hot_labels and negative_one_hot_labels
            are the one-hot encoded positive and negative labels.
        """
        x, y_pos = self.dataset[index]

        y_neg = y_pos
        while y_neg == y_pos:
            y_neg = torch.randint(0, NUM_CLASSES, (1,)).item()

        positive_one_hot_labels = torch.zeros(NUM_CLASSES)
        positive_one_hot_labels[y_pos] = 1.0

        negative_one_hot_labels = torch.zeros(NUM_CLASSES)
        negative_one_hot_labels[y_neg] = 1.0

        return ((x, x), (positive_one_hot_labels, negative_one_hot_labels))


class CustomTestDataset(Dataset):
    """
    A custom PyTorch Dataset for test data that wraps around another dataset 
    and applies custom processing.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to wrap around.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset and applies custom processing.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple (x, y), where x is the sample data and y is the label.
        """
        x, y = self.dataset[index]
        y = torch.tensor(y)

        return x, y


def train_collate_fn(batch):
    """
    Custom collate function for training data to be used in a DataLoader.

    Args:
        batch (list): A list of samples retrieved from the CustomTrainDataset.

    Returns:
        tuple: A tuple (TrainInputData, TrainLabelData), where TrainInputData and TrainLabelData
        are custom objects storing the batched data and labels.
    """

    # 1. Transform the batch from a list of tuples into a tuple of lists
    data_batch, label_batch = zip(*batch)

    # 2. Further separate the data and labels into positive and negative parts
    data1, data2 = zip(*data_batch)
    positive_labels, negative_labels = zip(*label_batch)

    # 3. Convert lists to tensors
    data1 = torch.stack(data1)
    data2 = torch.stack(data2)
    positive_labels = torch.stack(positive_labels)
    negative_labels = torch.stack(negative_labels)

    # 4. Repeat along a new dimension for ITERATIONS times
    data1 = data1.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    data2 = data2.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    positive_labels = positive_labels.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    negative_labels = negative_labels.unsqueeze(0).repeat(ITERATIONS, 1, 1)

    # 5. Return as custom objects
    return TrainInputData(data1, data2), TrainLabelData(positive_labels, negative_labels)


def test_collate_fn(batch):
    """
    Custom collate function for test data to be used in a DataLoader.

    Args:
        batch (list): A list of samples retrieved from the CustomTestDataset.

    Returns:
        TestData: A custom object storing the batched data and labels.
    """

    # 1. Transform the batch from a list of tuples into a tuple of lists
    data, labels = zip(*batch)

    # 2. Convert lists to tensors
    data = torch.stack(data)
    labels = torch.stack(labels)

    # 3. Repeat along a new dimension for ITERATIONS times
    data = data.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    labels = labels.unsqueeze(0).repeat(ITERATIONS, 1, 1)

    # 4. Return as a custom object
    return TestData(data, labels)


def MNIST_loaders(train_batch_size, test_batch_size):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(CustomTrainDataset(MNIST('./data/', train=True,
                                                       download=True,
                                                       transform=transform)),
                              batch_size=train_batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=8)

    test_loader = DataLoader(CustomTestDataset(MNIST('./data/', train=False,
                                                     download=True,
                                                     transform=transform)),
                             batch_size=test_batch_size, shuffle=False, collate_fn=test_collate_fn, num_workers=8)

    return train_loader, test_loader


def convert_to_timestep_dims(data):
    # Create a tensor of shape (1, data_size)
    data_unsqueezed = data.unsqueeze(0)
    # Create a tensor of shape (ITERATIONS, data_size)
    data_repeated = data_unsqueezed.repeat(ITERATIONS, 1)
    return data_repeated


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",

        # track hyperparameters and run metadata
        config={
            "architecture": "Recurrent-FF",
            "dataset": "MNIST",
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
