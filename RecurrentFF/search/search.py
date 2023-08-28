import random
from multiprocessing import Process

import torch
import wandb

from RecurrentFF.benchmarks.mnist.mnist import MNIST_loaders
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import DataConfig, Settings
from RecurrentFF.util import set_logging

DATA_SIZE = 784
NUM_CLASSES = 10
TRAIN_BATCH_SIZE = 500
TEST_BATCH_SIZE = 5000

DEVICE = "mps"


def run(settings: Settings):
    # Needs to be done here as well because multiprocessing.
    set_logging()

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    print(settings.model_dump())

    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",

        # track hyperparameters and run metadata
        config={
            "architecture": "Recurrent-FF",
            "dataset": "MNIST",
            "settings": settings.model_dump(),
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)

    try:
        model.train(train_loader, test_loader)
        print("======================FINISHED RUN============================")
    except KeyboardInterrupt:
        print("======================FINISHED RUN============================")
        exit(0)


if __name__ == "__main__":
    set_logging()

    loss_thresholds = [0.75, 1, 1.25]

    iterations = [10, 20]

    hidden_sizes = [[500, 500, 500], [1000, 1000, 1000]]

    ff_act = ["relu"]

    ff_optimizers = ["rmsprop", "adam"]
    classifier_optimizers = ["rmsprop", "adam"]

    ff_rmsprop_momentums = [0.0, 0.2, 0.5, 0.7, 0.9]
    ff_rmsprop_learning_rates = [0.00001, 0.00005, 0.0001, 0.0005]
    classifier_rmsprop_momentums = [0.0, 0.2, 0.5, 0.7, 0.9]
    classifier_rmsprop_learning_rates = [0.00001, 0.00005, 0.0001, 0.0005]

    ff_adam_learning_rates = [0.00001, 0.00005, 0.0001]
    classifier_adam_learning_rates = [0.00001, 0.00005, 0.0001, 0.0005]

    ff_adadelta_learning_rates = [0.00001, 0.00005, 0.0001, 0.0005]
    classifier_adadelta_learning_rates = [0.00001, 0.00005, 0.0001, 0.0005]

    train_batch_sizes = [200, 500, 1000, 2000]
    densities = [1]
    damping_factors = [0.3, 0.4, 0.5, 0.6, 0.7]

    while True:
        # random hyperparams
        loss_threshold = random.choice(loss_thresholds)
        iterations_ = random.choice(iterations)
        hidden_sizes_ = random.choice(hidden_sizes)
        act = random.choice(ff_act)
        ff_opt = random.choice(ff_optimizers)
        classifier_opt = random.choice(classifier_optimizers)
        ff_rmsprop_momentum = random.choice(ff_rmsprop_momentums)
        ff_rmsprop_learning_rate = random.choice(ff_rmsprop_learning_rates)
        classifier_rmsprop_momentum = random.choice(
            classifier_rmsprop_momentums)
        classifier_rmsprop_learning_rate = random.choice(
            classifier_rmsprop_learning_rates)
        ff_adam_learning_rate = random.choice(ff_adam_learning_rates)
        classifier_adam_learning_rate = random.choice(
            classifier_adam_learning_rates)
        train_batch_size = random.choice(train_batch_sizes)
        ff_adadelta_learning_rate = random.choice(ff_adadelta_learning_rates)
        classifier_adadelta_learning_rate = random.choice(
            classifier_adadelta_learning_rates)
        density = random.choice(densities)
        damping_factor = random.choice(damping_factors)

        # construct settings
        settings = Settings.new()

        settings.device.device = DEVICE

        data_config = {
            "data_size": DATA_SIZE,
            "num_classes": NUM_CLASSES,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "test_batch_size": TEST_BATCH_SIZE,
            "iterations": iterations_}

        if settings.data_config is None:
            settings.data_config = DataConfig(**data_config)

        # mutate settings
        settings.data_config.iterations = iterations_
        settings.data_config.train_batch_size = train_batch_size

        settings.model.loss_threshold = loss_threshold
        settings.model.hidden_sizes = hidden_sizes_
        settings.model.ff_activation = act
        settings.model.ff_optimizer = ff_opt
        settings.model.classifier_optimizer = classifier_opt
        settings.model.interconnect_density = density
        settings.model.damping_factor = damping_factor

        if ff_opt == "rmsprop":
            settings.model.ff_rmsprop.momentum = ff_rmsprop_momentum
            settings.model.ff_rmsprop.learning_rate = ff_rmsprop_learning_rate
        elif ff_opt == "adam":
            settings.model.ff_adam.learning_rate = ff_adam_learning_rate
        elif ff_opt == "adadelta":
            settings.model.ff_adadelta.learning_rate = ff_adadelta_learning_rate

        if classifier_opt == "rmsprop":
            settings.model.classifier_rmsprop.momentum = classifier_rmsprop_momentum
            settings.model.classifier_rmsprop.learning_rate = classifier_rmsprop_learning_rate
        elif classifier_opt == "adam":
            settings.model.classifier_adam.learning_rate = classifier_adam_learning_rate
        elif classifier_opt == "adadelta":
            settings.model.classifier_adadelta.learning_rate = classifier_adadelta_learning_rate

        # run hyperparams
        p = Process(target=run, args=(
            settings,
        ))
        p.start()
        p.join()
        print("-----------", str(p.exitcode))
        if p.exitcode != 0:
            exit(1)

        break
