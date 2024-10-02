from datetime import datetime
import logging
import random
import string
from typing import List, Tuple, cast
import torch
from torch import nn
import wandb

from RecurrentFF.model.data_scenario.processor import DataScenario
from RecurrentFF.model.data_scenario.static_single_class import (
    StaticSingleClassProcessor,
)
from RecurrentFF.model.hidden_layer import HiddenLayer, ResidualConnection, WeightInitialization
from RecurrentFF.model.inner_layers import InnerLayers, LayerMetrics
from RecurrentFF.util import (
    Activations,
    ForwardMode,
    LatentAverager,
    TrainInputData,
    TrainLabelData,
    TrainTestBridgeFormatLoader,
    layer_activations_to_badness,
    swap_top_two_softmax,
    is_confident,
    zero_correct_class_softmax,
    percent_above_threshold,
    percent_correct,
    sample_avoiding_correct_class,
    sample_from_logits
)
from RecurrentFF.settings import (
    Settings,
)


# TODO: try use separate optimizer for lateral connections
# TODO: try different learning rates for lateral connections
class RecurrentFFNet(nn.Module):
    """
    Implements a Recurrent Forward-Forward Network (RecurrentFFNet) based on
    PyTorch's nn.Module.

    This class represents a multi-layer network composed of an input layer, one
    or more hidden layers, and an output layer. Unlike traditional feed-forward
    networks, the hidden layers in this network are recurrent, i.e., they are
    connected back to themselves across timesteps.

    The learning procedure used here is a variant of the "Forward-Forward"
    algorithm, which is a greedy multi-layer learning method inspired by
    Boltzmann machines and Noise Contrastive Estimation. Instead of a
    traditional forward and backward pass, this algorithm employs two forward
    passes operating on different data and with contrasting objectives.

    During training, a "positive" pass operates on real input data and adjusts
    the weights to decrease the 'badness' in each hidden layer. The 'badness' is
    calculated as the sum of squared activation values. On the other hand, a
    "negative" pass operates on fake "negative data" and adjusts the weights to
    increase the 'badness' in each hidden layer.

    The hidden layers are instances of the HiddenLayer class. The hidden layers
    are connected to each other and the output layer, forming a fully connected
    recurrent architecture.
    """

    def __init__(self, settings: Settings):
        logging.info("Initializing network")
        super(RecurrentFFNet, self).__init__()

        self.settings = settings

        logging.info("Creating layers")
        inner_layers = nn.ModuleList()
        prev_size = self.settings.data_config.data_size
        for i, size in enumerate(self.settings.model.hidden_sizes):
            next_size = self.settings.model.hidden_sizes[i + 1] if i < len(
                self.settings.model.hidden_sizes) - 1 else self.settings.data_config.num_classes

            layer_num = i
            hidden_layer = HiddenLayer(
                self.settings,
                self.settings.data_config.train_batch_size,
                self.settings.data_config.test_batch_size,
                prev_size,
                size,
                next_size,
                self.settings.model.damping_factor,
                layer_num)
            inner_layers.append(hidden_layer)
            prev_size = size

        # attach layers to each other
        logging.info("Attaching layers")
        for i in range(1, len(inner_layers)):
            hidden_layer = inner_layers[i]
            hidden_layer.set_previous_layer(inner_layers[i - 1])

        for i in range(0, len(inner_layers) - 1):
            hidden_layer = inner_layers[i]
            hidden_layer.set_next_layer(inner_layers[i + 1])

        # initialize the residual connections
        logging.info("Attaching residual layer connections")
        for i in range(0, len(inner_layers)):
            for j in range(0, len(inner_layers)):
                # TODO: Perform testing for residual connections and determine best scheme. Examples:
                # if i != j and abs(i - j) != 1:
                # if abs(i-j) == 4:
                if False:
                    source = inner_layers[i]
                    target = inner_layers[j]
                    if i < j:
                        weight_init = WeightInitialization.Forward
                    else:
                        weight_init = WeightInitialization.Backward
                    residual_connection = ResidualConnection(
                        source, target.size, settings.model.dropout, weight_init)
                    inner_layers[i].init_residual_connection(
                        residual_connection)

        # initialize optimizers
        logging.info("Initializing optimizer")
        for layer in inner_layers:
            layer.init_optimizer()

        self.inner_layers = InnerLayers(self.settings, inner_layers)

        # when we eventually support changing/multiclass scenarios this will be
        # configurable
        logging.info("Initializing processor")
        self.processor = StaticSingleClassProcessor(
            self.inner_layers, self.settings)

        self.weights_file_name = self.settings.data_config.dataset + \
            "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + ''.join(
                random.choices(string.ascii_uppercase + string.digits, k=6)) + ".pth"

        logging.info("Constructing decoder")
        size = self.settings.model.hidden_sizes[-1]
        num_layers = len(self.settings.model.hidden_sizes)
        generative_size = size * num_layers
        self.generative_linear = torch.nn.Sequential(
            # nn.Linear(generative_size, generative_size),
            # nn.ReLU(),
            # nn.Linear(generative_size, size),
            # nn.ReLU(),
            nn.Linear(generative_size, settings.data_config.data_size +
                      settings.data_config.num_classes)
        )
        for layer in self.generative_linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        self.optimizer = torch.optim.Adam(
            self.generative_linear.parameters(), lr=self.settings.model.ff_rmsprop.learning_rate)

        logging.info("Finished initializing network")

    def predict(
            self,
            data_scenario: DataScenario,
            data_loader: torch.utils.data.DataLoader,
            num_batches: int,
            write_activations: bool = False) -> None:
        self.eval()
        if data_scenario == DataScenario.StaticSingleClass:
            self.processor.brute_force_predict(
                data_loader,
                num_batches,
                is_test_set=True,
                write_activations=write_activations)

    def train_model(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader) -> None:
        """
        Trains the RecurrentFFNet model using the provided train and test data loaders.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader providing the training data and labels.
            test_loader (torch.utils.data.DataLoader): DataLoader providing the test data and labels.

        Procedure:
            For each epoch, the method iterates over batches from the train_loader. For each batch, it resets
            the network's activations and performs a preinitialization step, forwarding both positive and negative
            data through the network. It then runs a specified number of iterations, where it trains the network
            using the input and label data.

            After each batch, the method calculates the 'badness' metric for each layer in the network (for both
            positive and negative data), averages the layer losses, and then calculates the prediction accuracy on
            the test data using test_loader.

            Finally, the method logs these metrics (accuracy, average loss, and layer-wise 'badness' scores)
            for monitoring the training process. The metrics logged depend on the number of layers in the network.

        Note:
            'Badness' here refers to a metric that indicates how well the model's current activations represent
            a given class. It's calculated by a function `layer_activations_to_badness`, which transforms a
            layer's activations into a 'badness' score. This function operates on the RecurrentFFNet model level
            and is called during the training process.
        """

        total_batch_count = 0
        best_test_accuracy: float = 0
        confidence_threshold = {"value": 0.01}
        for epoch in range(0, self.settings.model.epochs):
            logging.info("Epoch: " + str(epoch))
            self.train()

            for batch_num, (input_data, label_data) in enumerate(train_loader):
                input_data.move_to_device_inplace(self.settings.device.device)
                label_data.move_to_device_inplace(self.settings.device.device)

                if self.settings.model.should_replace_neg_data:
                    self.processor.replace_negative_data_inplace(
                        input_data.pos_input, label_data, total_batch_count)

                layer_metrics, pos_badness_per_layer, neg_badness_per_layer = self.__train_batch(
                    epoch, batch_num, input_data, label_data, total_batch_count, confidence_threshold=confidence_threshold)

                if self.settings.model.should_log_metrics:
                    self.__log_batch_metrics(
                        layer_metrics,
                        pos_badness_per_layer,
                        neg_badness_per_layer,
                        total_batch_count)

                total_batch_count += 1

            self.eval()

            # TODO: make train batches equal to however much a single test
            # batch is w.r.t. total samples
            #
            # TODO: Fix this hacky data loader bridge format
            test_accuracy = self.processor.brute_force_predict(
                test_loader, self.generative_linear, self.optimizer, 1, True)
            train_accuracy = self.processor.brute_force_predict(
                TrainTestBridgeFormatLoader(train_loader), self.generative_linear, self.optimizer, 1, False)  # type: ignore[arg-type]

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(self.state_dict(), self.weights_file_name)

            if self.settings.model.should_log_metrics:
                self.__log_epoch_metrics(
                    train_accuracy,
                    test_accuracy,
                    epoch,
                    total_batch_count
                )

            self.inner_layers.step_learning_rates()

    # def __train_batch(
    #             self,
    #             epoch_num: int,
    #             batch_num: int,
    #             input_data: TrainInputData,
    #             label_data: TrainLabelData,
    #             total_batch_count: int,
    #             confidence_threshold: dict) -> Tuple[LayerMetrics, List[float], List[float]]:
    #         logging.info(f"Batch: {batch_num}")

    #         # First cycle with original dataset
    #         self.inner_layers.reset_activations(True)

    #         preinit_upper_clamped_tensor = self.processor.get_preinit_upper_clamped_tensor(
    #             label_data.pos_labels[0].shape)
    #         for preinit_step in range(self.settings.model.prelabel_timesteps):
    #             logging.debug(f"Preinitialization step: {preinit_step}")

    #             pos_input = input_data.pos_input[0]
    #             neg_input = input_data.neg_input[0]

    #             self.inner_layers.advance_layers_forward(
    #                 ForwardMode.PositiveData, pos_input, preinit_upper_clamped_tensor, False)
    #             self.inner_layers.advance_layers_forward(
    #                 ForwardMode.NegativeData, neg_input, preinit_upper_clamped_tensor, False)

    #         num_layers = len(self.settings.model.hidden_sizes)
    #         layer_metrics = LayerMetrics(num_layers)

    #         pos_badness_per_layer = []
    #         neg_badness_per_layer = []
    #         iterations = input_data.pos_input.shape[0]
    #         batch_size = input_data.pos_input.shape[1]
    #         pos_target_latents_averager = LatentAverager()
    #         class_predictions_agg = torch.zeros(batch_size, self.settings.data_config.num_classes).to(self.settings.device.device)

    #         incorrect_count = torch.zeros(batch_size, dtype=torch.int).to(self.settings.device.device)

    #         for iteration in range(iterations):
    #             logging.debug(f"Iteration: {iteration}")

    #             data_criterion = torch.nn.MSELoss()
    #             label_criterion = torch.nn.CrossEntropyLoss()
    #             generative_input = torch.zeros(batch_size, self.settings.data_config.data_size + self.settings.data_config.num_classes).to(self.settings.device.device)
    #             assert not generative_input.requires_grad

    #             for layer in self.inner_layers:
    #                 layer.optimizer.zero_grad()
    #                 activations = layer.pos_activations.current
    #                 generative_input += layer.generative_linear(activations)

    #             reconstructed_data, reconstructed_labels = generative_input.split(
    #                 [self.settings.data_config.data_size, self.settings.data_config.num_classes], dim=1)

    #             data_loss = data_criterion(reconstructed_data, input_data.pos_input[iteration])
    #             label_loss = label_criterion(reconstructed_labels, torch.argmax(label_data.pos_labels[iteration], dim=1))
    #             loss = data_loss + label_loss

    #             wandb.log({
    #                 "generative loss": loss.item(),
    #                 "data loss": data_loss.item(),
    #                 "label loss": label_loss.item()
    #             }, step=total_batch_count)

    #             loss.backward()

    #             for layer in self.inner_layers:
    #                 assert layer.forward_linear.weight.grad is None or torch.all(layer.forward_linear.weight.grad == 0)
    #                 layer.optimizer.step()

    #             generative_input = generative_input.detach()

    #             input_data_sample = (input_data.pos_input[iteration], input_data.pos_input[iteration])
    #             label_data_sample = (
    #                 torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1),
    #                 zero_correct_class_softmax(generative_input[:, self.settings.data_config.data_size:], label_data.pos_labels[iteration]),
    #             )

    #             class_predictions_agg += torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1)

    #             self.inner_layers.advance_layers_train(
    #                 input_data_sample, label_data_sample, True, layer_metrics)

    #             lower_iteration_threshold = iterations // 2 - iterations // 10
    #             upper_iteration_threshold = iterations // 2 + iterations // 10

    #             if lower_iteration_threshold <= iteration <= upper_iteration_threshold:
    #                 pos_badness_per_layer.append([layer_activations_to_badness(
    #                     cast(Activations, layer.pos_activations).current).mean() for layer in self.inner_layers])
    #                 neg_badness_per_layer.append([layer_activations_to_badness(
    #                     cast(Activations, layer.neg_activations).current).mean() for layer in self.inner_layers])

    #                 positive_latents = [cast(Activations, layer.pos_activations).current for layer in self.inner_layers]
    #                 positive_latents_collapsed = torch.cat(positive_latents, dim=1)
    #                 pos_target_latents_averager.track_collapsed_latents(positive_latents_collapsed)

    #             percent_c = percent_correct(torch.softmax(reconstructed_labels, dim=1), label_data.pos_labels[iteration])
    #             percent_above = percent_above_threshold(torch.softmax(reconstructed_labels, dim=1), label_data.pos_labels[iteration], 0.5)
    #             conf, should_stop = is_confident(torch.softmax(reconstructed_labels, dim=1), label_data.pos_labels[iteration], confidence_threshold["value"])

    #             # Count incorrect predictions
    #             predicted_labels = torch.argmax(torch.softmax(reconstructed_labels, dim=1), dim=1)
    #             true_labels = torch.argmax(label_data.pos_labels[iteration], dim=1)
    #             incorrect_mask = predicted_labels != true_labels
    #             incorrect_count += incorrect_mask

    #         # determine accuracy from class aggregations
    #         correct_percent_agg = (torch.argmax(class_predictions_agg, dim=1) == torch.argmax(label_data.pos_labels[0], dim=1)).float().mean().item() * 100

    #         wandb.log({
    #             "percent_correct": correct_percent_agg,
    #             "percent_above": percent_above,
    #             "avg_confidence_correct_class": conf,
    #             "confidence_threshold": confidence_threshold["value"]
    #         }, step=total_batch_count)

    #         # Create new dataset with samples that were incorrect at least 4 times
    #         new_input_data = []
    #         new_label_data = []
    #         total_incorrect_samples = 0
    #         for sample_idx in range(batch_size):
    #             if incorrect_count[sample_idx] >= 4:
    #                 total_incorrect_samples += 1
    #                 for iteration in range(iterations):
    #                     new_input_data.append(input_data.pos_input[iteration][sample_idx])
    #                     new_label_data.append(label_data.pos_labels[iteration][sample_idx])
    #         percent_incorrect = total_incorrect_samples / batch_size
    #         wandb.log({"percent_incorrect": percent_incorrect}, step=total_batch_count)

    #         # If no samples were incorrect at least 4 times, use the original dataset
    #         if not new_input_data:
    #             new_input_data = input_data.pos_input.reshape(-1, input_data.pos_input.shape[2])
    #             new_label_data = label_data.pos_labels.reshape(-1, label_data.pos_labels.shape[2])
    #         else:
    #             new_input_data = torch.stack(new_input_data)
    #             new_label_data = torch.stack(new_label_data)

    #         # Duplicate samples to match the original dataset size if necessary
    #         while len(new_input_data) < iterations * batch_size:
    #             new_input_data = torch.cat([new_input_data, new_input_data[:iterations * batch_size - len(new_input_data)]])
    #             new_label_data = torch.cat([new_label_data, new_label_data[:iterations * batch_size - len(new_label_data)]])

    #         new_input_data = new_input_data.reshape(iterations, batch_size, -1)
    #         new_label_data = new_label_data.reshape(iterations, batch_size, -1)

    #         # Second cycle with new dataset
    #         self.inner_layers.reset_activations(True)

    #         preinit_upper_clamped_tensor = self.processor.get_preinit_upper_clamped_tensor(new_label_data[0].shape)
    #         for preinit_step in range(self.settings.model.prelabel_timesteps):
    #             logging.debug(f"Preinitialization step (new dataset): {preinit_step}")

    #             pos_input = new_input_data[0]
    #             neg_input = new_input_data[0]  # Using the same input for negative as we don't have separate negative data

    #             self.inner_layers.advance_layers_forward(
    #                 ForwardMode.PositiveData, pos_input, preinit_upper_clamped_tensor, False)
    #             self.inner_layers.advance_layers_forward(
    #                 ForwardMode.NegativeData, neg_input, preinit_upper_clamped_tensor, False)

    #         for iteration in range(iterations):
    #             logging.debug(f"Iteration (new dataset): {iteration}")

    #             data_criterion = torch.nn.MSELoss()
    #             label_criterion = torch.nn.CrossEntropyLoss()
    #             generative_input = torch.zeros(batch_size, self.settings.data_config.data_size + self.settings.data_config.num_classes).to(self.settings.device.device)
    #             assert not generative_input.requires_grad

    #             for layer in self.inner_layers:
    #                 layer.optimizer.zero_grad()
    #                 activations = layer.pos_activations.current
    #                 generative_input += layer.generative_linear(activations)

    #             reconstructed_data, reconstructed_labels = generative_input.split(
    #                 [self.settings.data_config.data_size, self.settings.data_config.num_classes], dim=1)

    #             data_loss = data_criterion(reconstructed_data, new_input_data[iteration])
    #             label_loss = label_criterion(reconstructed_labels, torch.argmax(new_label_data[iteration], dim=1))
    #             loss = data_loss + label_loss

    #             loss.backward()

    #             for layer in self.inner_layers:
    #                 assert layer.forward_linear.weight.grad is None or torch.all(layer.forward_linear.weight.grad == 0)
    #                 layer.optimizer.step()

    #             generative_input = generative_input.detach()

    #             input_data_sample = (new_input_data[iteration], new_input_data[iteration])
    #             label_data_sample = (
    #                 torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1),
    #                 zero_correct_class_softmax(generative_input[:, self.settings.data_config.data_size:], new_label_data[iteration]),
    #             )

    #             class_predictions_agg += torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1)

    #             self.inner_layers.advance_layers_train(
    #                 input_data_sample, label_data_sample, True, layer_metrics)

    #         if self.settings.model.should_replace_neg_data:
    #             pos_target_latents = pos_target_latents_averager.retrieve()
    #             self.processor.train_class_predictor_from_latents(
    #                 pos_target_latents, new_label_data[0], total_batch_count)

    #         pos_badness_per_layer_condensed: List[float] = [
    #             sum(layer_badnesses) / len(layer_badnesses) for layer_badnesses in zip(*pos_badness_per_layer)]
    #         neg_badness_per_layer_condensed: List[float] = [
    #             sum(layer_badnesses) / len(layer_badnesses) for layer_badnesses in zip(*neg_badness_per_layer)]

    #         return layer_metrics, pos_badness_per_layer_condensed, neg_badness_per_layer_condensed

    def __train_batch(
            self,
            epoch_num: int,
            batch_num: int,
            input_data: TrainInputData,
            label_data: TrainLabelData,
            total_batch_count: int,
            confidence_threshold: dict) -> Tuple[LayerMetrics, List[float], List[float]]:
        logging.info("Batch: " + str(batch_num))

        self.inner_layers.reset_activations(True)

        preinit_upper_clamped_tensor = self.processor.get_preinit_upper_clamped_tensor(
            label_data.pos_labels[0].shape)
        for preinit_step in range(0, self.settings.model.prelabel_timesteps):
            logging.debug("Preinitialization step: " +
                          str(preinit_step))

            pos_input = input_data.pos_input[0]
            neg_input = input_data.neg_input[0]

            self.inner_layers.advance_layers_forward(
                ForwardMode.PositiveData, pos_input, preinit_upper_clamped_tensor, False)
            self.inner_layers.advance_layers_forward(
                ForwardMode.NegativeData, neg_input, preinit_upper_clamped_tensor, False)

        num_layers = len(self.settings.model.hidden_sizes)
        layer_metrics = LayerMetrics(num_layers)

        pos_badness_per_layer = []
        neg_badness_per_layer = []
        iterations = input_data.pos_input.shape[0]
        pos_target_latents_averager = LatentAverager()
        class_predictions_agg = torch.zeros(
            input_data.pos_input[0].shape[0], self.settings.data_config.num_classes).to(self.settings.device.device)
        for iteration in range(0, iterations):
            logging.debug("Iteration: " + str(iteration))

            data_criterion = torch.nn.MSELoss()
            label_criterion = torch.nn.CrossEntropyLoss()
            # generative_input = torch.zeros(self.settings.data_config.train_batch_size, self.settings.data_config.data_size +
            #                                self.settings.data_config.num_classes).to(self.settings.device.device)
            # assert generative_input.requires_grad == False
            # for layer in self.inner_layers:
            #     layer.optimizer.zero_grad()
            #     activations = layer.pos_activations.current
            #     generative_input += layer.generative_linear(activations)
            self.optimizer.zero_grad()
            generative_input = self.generative_linear(
                torch.cat([layer.pos_activations.current for layer in self.inner_layers], dim=1)
            )
            assert generative_input.shape[0] == input_data.pos_input[iteration].shape[0] and generative_input.shape[
                1] == input_data.pos_input[iteration].shape[1] + self.settings.data_config.num_classes
            reconstructed_data, reconstructed_labels = generative_input.split(
                [self.settings.data_config.data_size, self.settings.data_config.num_classes], dim=1)
            if random.randint(0, 50) == 1:
                is_correct = torch.argmax(label_data.pos_labels[iteration][0]) == torch.argmax(
                    torch.softmax(reconstructed_labels[0], dim=0))
                print(is_correct.item())
            # data_loss = data_criterion(
            #     reconstructed_data, input_data.pos_input[iteration])
            label_loss = label_criterion(reconstructed_labels, torch.argmax(
                label_data.pos_labels[iteration], dim=1))
            # if epoch_num < 5:
            #     label_loss = label_criterion(reconstructed_labels, torch.argmax(label_data.pos_labels[iteration], dim=1))
            # else:
            #     label_loss = label_criterion(reconstructed_labels, torch.argmax(generative_input[:, self.settings.data_config.data_size:], dim=1))
            # loss = data_loss + label_loss
            loss = label_loss
            wandb.log({"generative loss": loss.item()}, step=total_batch_count)
            # wandb.log({"data loss": data_loss.item()}, step=total_batch_count)
            wandb.log({"label loss": label_loss.item()},
                      step=total_batch_count)
            loss.backward()
            self.optimizer.step()
            for layer in self.inner_layers:
                layer.optimizer.step()
            for layer in self.inner_layers:
                layer.pos_activations.current = layer.pos_activations.current.clone().detach()
                layer.neg_activations.current = layer.neg_activations.current.clone().detach()
                layer.pos_activations.previous = layer.pos_activations.previous.clone().detach()
                layer.neg_activations.previous = layer.neg_activations.previous.clone().detach()

            # for layer in self.inner_layers:
            #     # assert not torch.all(layer.generative_linear.weight.grad == 0)
            #     assert layer.forward_linear.weight.grad == None or torch.all(
            #         layer.forward_linear.weight.grad == 0)
            #     layer.optimizer.step()
            # if epoch_num < 5:
            #     for layer in self.inner_layers:
            #         # assert not torch.all(layer.generative_linear.weight.grad == 0)
            #         assert layer.forward_linear.weight.grad == None or torch.all(layer.forward_linear.weight.grad == 0)
            #         layer.optimizer.step()
            # else:
            #     for layer in self.inner_layers:
            #         layer.optimizer.step()
            generative_input = generative_input.detach()

            input_data_sample = (
                input_data.pos_input[iteration],
                generative_input[:, 0:self.settings.data_config.data_size])
                # input_data.pos_input[iteration])
            label_data_sample = (
                torch.zeros(self.settings.data_config.train_batch_size, self.settings.data_config.num_classes).to(
                    self.settings.device.device),
                torch.zeros(self.settings.data_config.train_batch_size, self.settings.data_config.num_classes).to(
                    self.settings.device.device),
                # torch.softmax(
                #     generative_input[:, self.settings.data_config.data_size:], dim=1),
                # torch.softmax(
                #     generative_input[:, self.settings.data_config.data_size:], dim=1),
                # sample_avoiding_correct_class(
                #     generative_input[:, self.settings.data_config.data_size:],
                #     label_data.pos_labels[iteration]),
                # sample_from_logits(
                #     torch.softmax(
                #         generative_input[:, self.settings.data_config.data_size:], dim=1)
                # ),
                # label_data.pos_labels[iteration],
                # torch.softmax(
                #     generative_input[:, self.settings.data_config.data_size:], dim=1),
                # torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1),
                # swap_top_two_softmax(torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1))
                # zero_correct_class_softmax(
                #     generative_input[:, self.settings.data_config.data_size:], label_data.pos_labels[iteration]),
            )
            # if (batch_num + epoch_num) % 2 == 0:
            #     label_data_sample = (
            #         torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1),
            #         zero_correct_class_softmax(generative_input[:, self.settings.data_config.data_size:], label_data.pos_labels[iteration]),
            #         )
            # else:
            #     label_data_sample = (
            #         torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1),
            #         swap_top_two_softmax(torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1)),
            #         )

            # class_predictions.append(torch.softmax(generative_input[:, self.settings.data_config.data_size:], dim=1))
            class_predictions_agg += torch.softmax(
                generative_input[:, self.settings.data_config.data_size:], dim=1)

            self.inner_layers.advance_layers_train(
                input_data_sample, label_data_sample, True, layer_metrics)

            lower_iteration_threshold = iterations // 2 - \
                iterations // 10
            upper_iteration_threshold = iterations // 2 + \
                iterations // 10
            # lower_iteration_threshold = 0
            # upper_iteration_threshold = iterations

            if iteration >= lower_iteration_threshold and \
                    iteration <= upper_iteration_threshold:
                pos_badness_per_layer.append([layer_activations_to_badness(
                    cast(Activations, layer.pos_activations).current).mean() for layer in self.inner_layers])
                neg_badness_per_layer.append([layer_activations_to_badness(
                    cast(Activations, layer.neg_activations).current).mean() for layer in self.inner_layers])

                positive_latents = [
                    cast(Activations, layer.pos_activations).current for layer in self.inner_layers]
                positive_latents_collapsed = torch.cat(positive_latents, dim=1)
                pos_target_latents_averager.track_collapsed_latents(
                    positive_latents_collapsed)

            percent_c = percent_correct(torch.softmax(
                reconstructed_labels, dim=1), label_data.pos_labels[iteration])
            percent_above = percent_above_threshold(torch.softmax(
                reconstructed_labels, dim=1), label_data.pos_labels[iteration], 0.5)
            conf, should_stop = is_confident(torch.softmax(
                reconstructed_labels, dim=1), label_data.pos_labels[iteration], confidence_threshold["value"])

            # if it was over 90% confident in correct answer on average return
            # if should_stop:
            #     print(conf)
            #     print(confidence_threshold["value"])
            #     print(torch.softmax(reconstructed_labels, dim=1)[0:3])
            #     input()
            #
            # baseline_conf = 0.01
            # if should_stop and iteration > lower_iteration_threshold:
            #     confidence_threshold["value"] += 0.001
            #     print(iteration)
            #     break
            # elif iteration > 3 and confidence_threshold["value"] > baseline_conf:
            #     confidence_threshold["value"] -= 0.001

        # determine accuracy from class aggregations
        correct_percent_agg = (torch.argmax(class_predictions_agg, dim=1) == torch.argmax(
            label_data.pos_labels[0], dim=1)).float().mean().item() * 100

        wandb.log({"percent_correct": correct_percent_agg},
                  step=total_batch_count)
        wandb.log({"percent_above": percent_above}, step=total_batch_count)
        wandb.log({"avg_confidence_correct_class": conf},
                  step=total_batch_count)
        wandb.log(
            {"confidence_threshold": confidence_threshold["value"]}, step=total_batch_count)

        if self.settings.model.should_replace_neg_data:
            pos_target_latents = pos_target_latents_averager.retrieve()
            self.processor.train_class_predictor_from_latents(
                pos_target_latents, label_data.pos_labels[0], total_batch_count)

        pos_badness_per_layer_condensed: list[float] = [
            sum(layer_badnesses) /
            len(layer_badnesses) for layer_badnesses in zip(
                *
                pos_badness_per_layer)]
        neg_badness_per_layer_condensed: list[float] = [
            sum(layer_badnesses) /
            len(layer_badnesses) for layer_badnesses in zip(
                *
                neg_badness_per_layer)]

        return layer_metrics, pos_badness_per_layer_condensed, neg_badness_per_layer_condensed

    def __log_epoch_metrics(
            self,
            train_accuracy: float,
            test_accuracy: float,
            epoch: int,
            total_batch_count: int) -> None:
        wandb.log({"train_acc": train_accuracy,
                   "test_acc": test_accuracy,
                   "epoch": epoch}, step=total_batch_count)

    def __log_batch_metrics(
            self,
            layer_metrics: LayerMetrics,
            pos_badness_per_layer: List[float],
            neg_badness_per_layer: List[float],
            total_batch_count: int) -> None:
        # Supports wandb tracking of max 3 layer badnesses
        try:
            first_layer_pos_badness = pos_badness_per_layer[0]
            first_layer_neg_badness = neg_badness_per_layer[0]
            second_layer_pos_badness = pos_badness_per_layer[1]
            second_layer_neg_badness = neg_badness_per_layer[1]
            third_layer_pos_badness = pos_badness_per_layer[2]
            third_layer_neg_badness = neg_badness_per_layer[2]
        except BaseException:
            # No-op as there may not be 3 layers
            pass

        layer_metrics.log_metrics(total_batch_count)
        average_layer_loss = layer_metrics.average_layer_loss()

        if len(self.inner_layers) >= 3:
            wandb.log({"loss": average_layer_loss,
                       "first_layer_pos_badness": first_layer_pos_badness,
                       "second_layer_pos_badness": second_layer_pos_badness,
                       "third_layer_pos_badness": third_layer_pos_badness,
                       "first_layer_neg_badness": first_layer_neg_badness,
                       "second_layer_neg_badness": second_layer_neg_badness,
                       "third_layer_neg_badness": third_layer_neg_badness,
                       "batch": total_batch_count},
                      step=total_batch_count)
        elif len(self.inner_layers) == 2:
            wandb.log({
                "loss": average_layer_loss,
                "first_layer_pos_badness": first_layer_pos_badness,
                "second_layer_pos_badness": second_layer_pos_badness,
                "first_layer_neg_badness": first_layer_neg_badness,
                "second_layer_neg_badness": second_layer_neg_badness,
                "batch": total_batch_count},
                step=total_batch_count)

        elif len(self.inner_layers) == 1:
            wandb.log({
                "loss": average_layer_loss,
                "first_layer_pos_badness": first_layer_pos_badness,
                "first_layer_neg_badness": first_layer_neg_badness,
                "batch": total_batch_count},
                step=total_batch_count)
