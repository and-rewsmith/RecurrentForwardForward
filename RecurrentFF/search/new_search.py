import logging
import random
import time
from typing import TextIO
import torch
import wandb
from RecurrentFF.benchmarks.mnist.mnist import MNIST_loaders
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import Settings

from RecurrentFF.util import set_logging


# DATA_SIZE = 784
# NUM_CLASSES = 10
# TRAIN_BATCH_SIZE = 500
# TEST_BATCH_SIZE = 500

DEVICE = "mps"

NUM_SEEDS_BENCH = 1

datetime_str = time.strftime("%Y%m%d-%H%M%S")
RUNNING_LOG_FILENAME = f"running_log_{datetime_str}.log"


def objective() -> None:
    wandb.init(
        project="Recurrent-FF",
        config={
            "architecture": "Recurrent-FF",
            "dataset": "MNIST",
        },
        # allow_val_change=True  # TODOPRE: review this as it is silencing a warning
    )

    settings = Settings.new()

    settings.model.hidden_sizes = [100, 100, 100, 100, 100]

    settings.model.ff_rmsprop.learning_rate = wandb.config.learning_rate
    settings.model.ff_rmsprop.momentum = wandb.config.momentum

    settings.model.prelabel_timesteps = wandb.config.prelabel_timesteps
    settings.data_config.iterations = wandb.config.iterations

    settings.model.loss_scale_ff = wandb.config.loss_scale_ff
    settings.model.loss_scale_predictive = wandb.config.loss_scale_predictive
    settings.model.loss_scale_hebbian = wandb.config.loss_scale_hebbian
    settings.model.loss_scale_decorrelative = wandb.config.loss_scale_decorrelative

    settings.model.damping_factor = wandb.config.damping_factor

    # layer_sizes = wandb.config.layer_sizes
    # learning_rate = wandb.config.learning_rate
    # dt = wandb.config.dt
    # exc_to_inhib_conn_c = wandb.config.exc_to_inhib_conn_c
    # exc_to_inhib_conn_sigma_squared = wandb.config.exc_to_inhib_conn_sigma_squared
    # percentage_inhibitory = wandb.config.percentage_inhibitory
    # decay_beta = wandb.config.decay_beta
    # tau_mean = wandb.config.tau_mean
    # tau_var = wandb.config.tau_var
    # tau_stdp = wandb.config.tau_stdp
    # tau_rise_alpha = wandb.config.tau_rise_alpha
    # tau_fall_alpha = wandb.config.tau_fall_alpha
    # tau_rise_epsilon = wandb.config.tau_rise_epsilon
    # tau_fall_epsilon = wandb.config.tau_fall_epsilon
    # threshold_scale = wandb.config.threshold_scale
    # threshold_decay = wandb.config.threshold_decay

    # sum layer sizes to get total neurons
    # total_neurons = sum(layer_sizes)
    # layer_sparsity = NUM_NEURONS_CONNECT_ACROSS_LAYERS / total_neurons

    run_settings = f"""
    running with:
    settings: {settings.model_dump()}
    """
    # run_settings = f"""
    # running with:
    # layer_sizes: {layer_sizes}
    # learning_rate: {learning_rate}
    # dt: {dt}
    # percentage_inhibitory: {percentage_inhibitory}
    # exc_to_inhib_conn_c: {exc_to_inhib_conn_c}
    # exc_to_inhib_conn_sigma_squared: {exc_to_inhib_conn_sigma_squared}
    # layer_sparsity: {layer_sparsity}
    # decay_beta: {decay_beta},
    # tau_mean: {tau_mean},
    # tau_var: {tau_var},
    # tau_stdp: {tau_stdp},
    # tau_rise_alpha: {tau_rise_alpha},
    # tau_fall_alpha: {tau_fall_alpha},
    # tau_rise_epsilon: {tau_rise_epsilon},
    # tau_fall_epsilon: {tau_fall_epsilon},
    # threshold_scale: {threshold_scale},
    # threshold_decay: {threshold_decay},
    # """
    # logging.info(run_settings)

    with open(RUNNING_LOG_FILENAME, "a") as running_log:
        running_log.write(f"{run_settings}")
        running_log.flush()

        cum_pass_rate = 0
        for _ in range(NUM_SEEDS_BENCH):
            pass_rate = bench_specific_seed(
                running_log,
                settings
            )
            wandb.log({"train_accuracy": pass_rate})
            cum_pass_rate += pass_rate

        running_log.write(
            run_settings + f"train_accuracy: {cum_pass_rate / NUM_SEEDS_BENCH}\n\n======================================\
                =========================================")
        running_log.flush()

    wandb.log({"average_image_predict_success": cum_pass_rate / NUM_SEEDS_BENCH})


# @profile(stdout=False, filename='baseline.prof',
#          skip=True)
def bench_specific_seed(running_log: TextIO,
                        settings: Settings
                        ) -> float:
    rand = random.randint(1000, 9999)
    torch.manual_seed(rand)
    running_log.write(f"Seed: {rand}\n")

    # settings = Settings(
    #     layer_sizes=layer_sizes,
    #     data_size=dataset.num_classes,
    #     batch_size=BATCH_SIZE,
    #     learning_rate=learning_rate,
    #     epochs=10,
    #     encode_spike_trains=ENCODE_SPIKE_TRAINS,
    #     dt=dt,
    #     percentage_inhibitory=percentage_inhibitory,
    #     exc_to_inhib_conn_c=exc_to_inhib_conn_c,
    #     exc_to_inhib_conn_sigma_squared=exc_to_inhib_conn_sigma_squared,
    #     layer_sparsity=layer_sparsity,
    #     decay_beta=decay_beta,
    #     tau_mean=tau_mean,
    #     tau_var=tau_var,
    #     tau_stdp=tau_stdp,
    #     tau_rise_alpha=tau_rise_alpha,
    #     tau_fall_alpha=tau_fall_alpha,
    #     tau_rise_epsilon=tau_rise_epsilon,
    #     tau_fall_epsilon=tau_fall_epsilon,
    #     threshold_scale=threshold_scale,
    #     threshold_decay=threshold_decay,
    #     device=torch.device(DEVICE)
    # )

    # train_dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    net = RecurrentFFNet(settings).to(settings.device.device)

    train_loader, test_loader = MNIST_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    train_accuracy = net.train_model(train_loader, test_loader)

    message = f"""---------------------------------
    train_accuracy: {train_accuracy}
    ---------------------------------
    """
    running_log.write(message)
    running_log.flush()
    logging.info(message)

    return train_accuracy


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=10, sci_mode=False)
    set_logging()

    running_log = open(RUNNING_LOG_FILENAME, "w")
    message = f"Sweep logs. Current datetime: {time.ctime()}\n"
    running_log.write(message)
    running_log.close()
    logging.debug(message)

    sweep_id = "and-rewsmith/Recurrent-FF/zsvxrw5i"
    wandb.agent(sweep_id, function=objective)
