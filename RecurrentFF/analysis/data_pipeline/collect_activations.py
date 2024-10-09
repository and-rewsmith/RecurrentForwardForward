import torch
import random
import numpy as np

from RecurrentFF.benchmarks.cifar10.cifar10 import DATA_SIZE, ITERATIONS, NUM_CLASSES, \
    TRAIN_BATCH_SIZE, DATASET
from RecurrentFF.benchmarks.cifar10.cifar10 import CIFAR10_loaders
from RecurrentFF.model.data_scenario.processor import DataScenario
from RecurrentFF.util import set_logging
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import Settings, DataConfig

TEST_BATCH_SIZE = TRAIN_BATCH_SIZE
NUM_BATCHES = 1000

WEIGHTS_PATH = "/home/localuser/Documents/projects/RecurrentForwardForward/CIFAR10_2024-10-08_15-13-57_B6VJTI.pth"

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

    settings.model.should_log_metrics = False

    set_logging()

    # Set seed for random, numpy, and PyTorch (both CPU and GPU)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For GPU-specific random numbers
    torch.cuda.manual_seed_all(seed)  # If you are using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    # Pytorch utils.
    # torch.autograd.set_detect_anomaly(True)

    # Generate train data.
    train_loader, test_loader = CIFAR10_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    # # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)
    model.load_state_dict(torch.load(
        WEIGHTS_PATH, map_location=settings.device.device))
    model.predict(DataScenario.StaticSingleClass,
                  test_loader, 1, write_activations=False)

    # _train_loader, test_loader_tmp = MNIST_loaders(
    #     settings.data_config.train_batch_size, 1000)
    # model.predict(DataScenario.StaticSingleClass,
    #               test_loader_tmp, 1, write_activations=False)

    # input("Does the accuracy look good?")

    # model.predict(DataScenario.StaticSingleClass,
    #               test_loader, NUM_BATCHES, write_activations=True)
