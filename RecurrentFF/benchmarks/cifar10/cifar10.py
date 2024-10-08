import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import wandb

from RecurrentFF.model.data_scenario.static_single_class import SingleStaticClassTestData
from RecurrentFF.util import TrainInputData, TrainLabelData, set_logging
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import Settings, DataConfig

DATA_SIZE = 3072  # 32x32x3
NUM_CLASSES = 10
TRAIN_BATCH_SIZE = 500
TEST_BATCH_SIZE = 500
ITERATIONS = 15
DATASET = "CIFAR10"

# If you want to load weights fill this in.
WEIGHTS_PATH = ""

PATCH_SIZE = 8


class CustomTrainDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
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
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y = torch.tensor(y)
        return x, y


def train_collate_fn(batch):
    data_batch, label_batch = zip(*batch)
    data1, data2 = zip(*data_batch)
    positive_labels, negative_labels = zip(*label_batch)

    data1 = torch.stack(data1)
    data2 = torch.stack(data2)
    positive_labels = torch.stack(positive_labels)
    negative_labels = torch.stack(negative_labels)

    data1 = data1.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    data2 = data2.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    positive_labels = positive_labels.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    negative_labels = negative_labels.unsqueeze(0).repeat(ITERATIONS, 1, 1)

    return TrainInputData(data1, data2), TrainLabelData(positive_labels, negative_labels)


def test_collate_fn(batch):
    data, labels = zip(*batch)

    data = torch.stack(data)
    labels = torch.stack(labels)

    data = data.unsqueeze(0).repeat(ITERATIONS, 1, 1)
    labels = labels.unsqueeze(0).repeat(ITERATIONS, 1)

    return SingleStaticClassTestData(data, labels)

def image_to_patches(img, patch_size):
    # img is a tensor of shape [C, H, W]
    # Get dimensions
    C, H, W = img.shape
    
    # Ensure the image height and width are divisible by the patch size
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
    
    # Calculate number of patches along each dimension
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Rearrange the image into patches and flatten each patch
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Now patches has shape [C, num_patches_h, num_patches_w, patch_size, patch_size]
    # Permute to bring patches together: [num_patches_h, num_patches_w, C, patch_size, patch_size]
    patches = patches.permute(1, 2, 0, 3, 4)
    # Reshape each patch to a flattened vector
    patches = patches.reshape(-1, C * patch_size * patch_size)
    return patches.flatten()

def CIFAR10_loaders(train_batch_size, test_batch_size):
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Lambda(lambda x: torch.flatten(x))
        Lambda(lambda x: image_to_patches(x, patch_size=PATCH_SIZE))
    ])

    train_loader = DataLoader(
        CustomTrainDataset(
            CIFAR10(
                './data/',
                train=True,
                download=True,
                transform=transform)),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=0,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

    test_loader = DataLoader(
        CustomTestDataset(
            CIFAR10(
                './data/',
                train=False,
                download=True,
                transform=transform)),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=0)

    return train_loader, test_loader


def convert_to_timestep_dims(data):
    data_unsqueezed = data.unsqueeze(0)
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

    torch.manual_seed(1234)

    wandb.init(
        project="Recurrent-FF",
        config={
            "architecture": "Recurrent-FF",
            "dataset": DATASET,
            "settings": settings.model_dump(),
        }
    )

    train_loader, test_loader = CIFAR10_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    model = RecurrentFFNet(settings).to(settings.device.device)

    if settings.model.should_load_weights:
        model.load_state_dict(torch.load(WEIGHTS_PATH))

    model.train_model(train_loader, test_loader)
