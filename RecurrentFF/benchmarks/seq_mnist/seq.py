import os

import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np

from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import Settings
from RecurrentFF.util import DataConfig, SingleStaticClassTestData, TrainInputData, TrainLabelData, set_logging

INPUT_SIZE = 4
NUM_CLASSES = 10
TRAIN_BATCH_SIZE = 5000
TEST_BATCH_SIZE = 5000
ITERATIONS = 150
FOCUS_ITERATION_NEG_OFFSET = 15
FOCUS_ITERATION_POS_OFFSET = 15


class SeqMnistTrainDataset(Dataset):
    """
    PyTorch Dataset class for loading data from tar files for the SeqMNIST task.

    Args:
        data_folder (str): Path to the folder containing the data files.
        transform (callable, optional): Optional transform to be applied on the data.
    """

    def __init__(self, data_folder, transform=None):
        super(SeqMnistTrainDataset, self).__init__()
        self.transform = transform
        self.path = data_folder

        # Get all the training files
        self.data_files = [
            f for f in os.listdir(
                os.path.join(
                    data_folder,
                    'sequences')) if f.startswith('trainimg-') and f.endswith('-targetdata.txt')]

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Tuple of inputs and labels. Inputs are a tuple of original data and
            its duplicate. Labels are a tuple of positive one-hot labels and negative one-hot labels.
        """
        data_file = self.data_files[index]

        # read data from file
        with open(os.path.join(self.path, 'sequences', data_file), 'r') as file:
            data = np.array([list(map(float, line.split()))
                            for line in file.read().splitlines()])

        x = torch.Tensor(data[:, 10:14])  # Your original tensor
        # repeat tensor rows to desired length
        x = self.repeat_data_to_length(x, ITERATIONS)

        # separate labels and data
        one_hot_labels = torch.Tensor(data[0, :10]).unsqueeze(
            0).repeat(data.shape[0], 1)
        one_hot_labels = self.repeat_data_to_length(
            one_hot_labels, ITERATIONS)  # repeat labels to desired length

        # create negative labels
        negative_one_hot_labels = self.get_negative_labels(
            one_hot_labels[0]).repeat(data.shape[0], 1)
        negative_one_hot_labels = self.repeat_data_to_length(
            negative_one_hot_labels, ITERATIONS)  # repeat negative labels to desired length

        if self.transform:
            x = self.transform(x)

        return ((x, x), (one_hot_labels, negative_one_hot_labels))

    def repeat_data_to_length(self, tensor, length):
        """
        Repeats the rows in a tensor until it reaches the desired length.

        Args:
            tensor (torch.Tensor): Tensor to be repeated.
            length (int): Desired length.

        Returns:
            torch.Tensor: Tensor repeated to the desired length.
        """
        repeat_factor = length // tensor.shape[0]
        remainder = length % tensor.shape[0]
        tensor = tensor.repeat(repeat_factor, 1)
        tensor = torch.cat([tensor, tensor[:remainder]], dim=0)
        return tensor

    def get_negative_labels(self, positive_labels):
        """
        Generates negative labels for a given set of positive labels.

        Args:
            positive_labels (Tensor): Tensor of positive labels.

        Returns:
            Tensor: Tensor of negative labels.
        """
        y_pos = positive_labels.argmax().item()
        y_neg = y_pos

        while y_neg == y_pos:
            y_neg = torch.randint(0, NUM_CLASSES, (1,)).item()

        # create negative one-hot label
        negative_one_hot_labels = torch.zeros_like(positive_labels)
        negative_one_hot_labels[y_neg] = 1

        return negative_one_hot_labels

    def __len__(self):
        """
        Calculates the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.data_files)


def collate_train(data):
    """
    Custom collate function to transpose the dimensions.

    Args:
        data (list): List of tuples containing data and labels.

    Returns:
        tuple: Tuple containing data and labels in the desired format.
    """
    data_batch, label_batch = zip(*data)

    pos_data, neg_data = zip(*data_batch)
    pos_labels, neg_labels = zip(*label_batch)

    # Convert to tensors
    pos_data = torch.stack(pos_data)
    neg_data = torch.stack(neg_data)
    pos_labels = torch.stack(pos_labels)
    neg_labels = torch.stack(neg_labels)

    # Transpose the tensors
    pos_data = pos_data.transpose(0, 1)
    neg_data = neg_data.transpose(0, 1)
    pos_labels = pos_labels.transpose(0, 1)
    neg_labels = neg_labels.transpose(0, 1)

    return TrainInputData(
        pos_data, neg_data), TrainLabelData(
        pos_labels, neg_labels)


class SeqMnistTestDataset(Dataset):
    """
    PyTorch Dataset class for loading data from tar files for the task.

    Args:
        data_folder (str): Path to the folder containing the data files.
        transform (callable, optional): Optional transform to be applied on the data.
    """

    def __init__(self, data_folder, transform=None):
        super(SeqMnistTestDataset, self).__init__()
        self.transform = transform
        self.path = data_folder

        # Get all the testing files
        self.data_files = [
            f for f in os.listdir(
                os.path.join(
                    data_folder,
                    'sequences')) if f.startswith('trainimg-') and f.endswith('-targetdata.txt')]

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Tuple containing data and label.
        """
        data_file = self.data_files[index]

        # read data from file
        with open(os.path.join(self.path, 'sequences', data_file), 'r') as file:
            data = np.array([list(map(float, line.split()))
                            for line in file.read().splitlines()])

        x = torch.Tensor(data[:, 10:14])  # Your original tensor
        # repeat tensor rows to desired length
        x = self.repeat_data_to_length(x, 150)

        # separate label and data
        one_hot_label = torch.Tensor(data[0, :10])
        y = one_hot_label.argmax()  # Convert one-hot label to scalar

        if self.transform:
            x = self.transform(x)

        return x, y

    def repeat_data_to_length(self, tensor, length):
        """
        Repeats the rows in a tensor until it reaches the desired length.

        Args:
            tensor (torch.Tensor): Tensor to be repeated.
            length (int): Desired length.

        Returns:
            torch.Tensor: Tensor repeated to the desired length.
        """
        repeat_factor = length // tensor.shape[0]
        remainder = length % tensor.shape[0]
        tensor = tensor.repeat(repeat_factor, 1)
        tensor = torch.cat([tensor, tensor[:remainder]], dim=0)
        return tensor

    def __len__(self):
        """
        Calculates the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.data_files)


def collate_test(batch):
    """
    Custom collate function to transpose the dimensions.

    Args:
        batch (list): List of tuples containing data and labels.

    Returns:
        tuple: Tuple containing data and labels in the desired format.
    """
    data, labels = zip(*batch)
    data = torch.stack(data).transpose(0, 1)
    labels = torch.stack(labels)

    return SingleStaticClassTestData(data, labels)


def MNIST_loaders(train_batch_size, test_batch_size):
    train_loader = DataLoader(
        SeqMnistTrainDataset('./RecurrentFF/benchmarks/Seq-MNIST/data/'),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_train,
        num_workers=8)

    test_loader = DataLoader(
        SeqMnistTestDataset('./RecurrentFF/benchmarks/Seq-MNIST/data/'),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_test,
        num_workers=8)

    return train_loader, test_loader


if __name__ == "__main__":
    settings = Settings()
    data_config = DataConfig(
        INPUT_SIZE,
        NUM_CLASSES,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        ITERATIONS,
        FOCUS_ITERATION_NEG_OFFSET,
        FOCUS_ITERATION_POS_OFFSET)

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
            "dataset": "Seq-MNIST",
            "epochs": settings.model.epochs,
            "learning_rate": settings.model.learning_rate,
            "layers": str(settings.model.hidden_sizes),
            "loss_threshold": settings.model.loss_threshold,
            "damping_factor": settings.model.damping_factor,
            "epsilon": settings.model.epsilon,
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    # Create and run model.
    model = RecurrentFFNet(data_config).to(settings.device.device)

    model.train(train_loader, test_loader)
