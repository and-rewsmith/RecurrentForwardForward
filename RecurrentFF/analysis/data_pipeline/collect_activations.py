import torch

from RecurrentFF.benchmarks.mnist.mnist import DATA_SIZE, ITERATIONS, NUM_CLASSES, \
    TRAIN_BATCH_SIZE, MNIST_loaders, DATASET
from RecurrentFF.model.data_scenario.processor import DataScenario
from RecurrentFF.util import set_logging
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import Settings, DataConfig

TEST_BATCH_SIZE = 1
NUM_BATCHES = 1000

WEIGHTS_PATH = ""

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

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)

    model.load_state_dict(torch.load(
        WEIGHTS_PATH, map_location=settings.device.device))

    _train_loader, test_loader_tmp = MNIST_loaders(
        settings.data_config.train_batch_size, 1000)
    model.predict(DataScenario.StaticSingleClass,
                  test_loader_tmp, 1, write_activations=False)

    input("Does the accuracy look good?")

    model.predict(DataScenario.StaticSingleClass,
                  test_loader, NUM_BATCHES, write_activations=True)
