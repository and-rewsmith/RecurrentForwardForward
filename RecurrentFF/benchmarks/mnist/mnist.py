import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import wandb

from RecurrentFF.model.data_scenario.static_single_class import SingleStaticClassTestData
from RecurrentFF.util import TrainInputData, TrainLabelData, set_logging
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import Settings, DataConfig

DATA_SIZE = 784
NUM_CLASSES = 10
TRAIN_BATCH_SIZE = 500
TEST_BATCH_SIZE = 5000
ITERATIONS = 15
DATASET = "MNIST"

# If you want to load weights fill this in.
WEIGHTS_PATH = ""


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
    return TrainInputData(
        data1, data2), TrainLabelData(
        positive_labels, negative_labels)


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
    labels = labels.unsqueeze(0).repeat(ITERATIONS, 1)

    # 4. Return as a custom object
    return SingleStaticClassTestData(data, labels)


def MNIST_loaders(train_batch_size, test_batch_size):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        CustomTrainDataset(
            MNIST(
                './data/',
                train=True,
                download=True,
                transform=transform)),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=0)

    test_loader = DataLoader(
        CustomTestDataset(
            MNIST(
                './data/',
                train=False,
                download=True,
                transform=transform)),
        batch_size=test_batch_size,
        shuffle=True,
        collate_fn=test_collate_fn,
        num_workers=0)

    return train_loader, test_loader


def convert_to_timestep_dims(data):
    # Create a tensor of shape (1, data_size)
    data_unsqueezed = data.unsqueeze(0)
    # Create a tensor of shape (ITERATIONS, data_size)
    data_repeated = data_unsqueezed.repeat(ITERATIONS, 1)
    return data_repeated


if __name__ == "__main__":
    settings = Settings.new()

    data_config = {
        "data_size": DATA_SIZE,
        "num_classes": NUM_CLASSES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "iterations": ITERATIONS,
        "dataset": DATASET}

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
            "dataset": DATASET,
            "settings": settings.model_dump(),
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)

    if settings.model.should_load_weights:
        model.load_state_dict(torch.load(WEIGHTS_PATH))

    model.train(train_loader, test_loader)
