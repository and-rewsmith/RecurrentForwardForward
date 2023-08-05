import logging


import torch
from torch import nn
import wandb
from profilehooks import profile

from RecurrentFF.model.data_scenario.static_single_class import (
    StaticSingleClassProcessor,
)
from RecurrentFF.model.hidden_layer import HiddenLayer
from RecurrentFF.model.inner_layers import InnerLayers
from RecurrentFF.util import (
    ForwardMode,
    OutputLayer,
    LatentAverager,
    layer_activations_to_goodness,
)
from RecurrentFF.settings import (
    Settings,
)


# TODO: store activations
# TODO: add conv layer at beginning to use receptive fields
# TODO: try sigmoid activation function
# TODO: use rms prop (emphasis on lateral connections)
# TODO: plumb optimizer into `HiddenLayer`
# TODO: different learning rates for lateral connections
# TODO: initialize weights (division by n, number of inputs) (lora activation)
# TODO: threshold a parameter?
# TODO: average activation
# TODO: look at Hinton norm
# TODO: log activations (variance is much bigger than average, then relu is not good - maybe try leaky relu?)
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
    the weights to increase the 'goodness' in each hidden layer. The 'goodness'
    is calculated as the sum of squared activation values. On the other hand, a
    "negative" pass operates on "negative data" and adjusts the weights to
    decrease the 'goodness' in each hidden layer.

    The hidden layers and output layer are instances of the HiddenLayer and
    OutputLayer classes, respectively. The hidden layers are connected to each
    other and the output layer, forming a fully connected recurrent
    architecture.
    """

    def __init__(self, data_config):
        logging.info("Initializing network")
        super(RecurrentFFNet, self).__init__()

        self.settings = Settings.new()
        self.data_config = data_config

        inner_layers = nn.ModuleList()
        prev_size = data_config.data_size
        for size in self.settings.model.hidden_sizes:
            hidden_layer = HiddenLayer(
                data_config.train_batch_size,
                data_config.test_batch_size,
                prev_size,
                size,
                self.settings.model.damping_factor)
            inner_layers.append(hidden_layer)
            prev_size = size

        self.output_layer = OutputLayer(
            self.settings.model.hidden_sizes[-1], self.data_config.num_classes)

        # attach layers to each other
        for i, hidden_layer in enumerate(inner_layers):
            if i != 0:
                hidden_layer.set_previous_layer(inner_layers[i - 1])

        for i, hidden_layer in enumerate(inner_layers):
            if i != len(inner_layers) - 1:
                hidden_layer.set_next_layer(inner_layers[i + 1])
            else:
                hidden_layer.set_next_layer(self.output_layer)

        self.inner_layers = InnerLayers(inner_layers)

        # when we eventually support changing/multiclass scenarios this will be
        # configurable
        self.processor = StaticSingleClassProcessor(
            self.inner_layers, data_config)

        logging.info("Finished initializing network")

    @profile(stdout=False, filename='baseline.prof',
             skip=Settings.new().model.skip_profiling)
    def train(self, train_loader, test_loader):
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

            After each batch, the method calculates the 'goodness' metric for each layer in the network (for both
            positive and negative data), averages the layer losses, and then calculates the prediction accuracy on
            the test data using test_loader.

            Finally, the method logs these metrics (accuracy, average loss, and layer-wise 'goodness' scores)
            for monitoring the training process. The metrics logged depend on the number of layers in the network.

        Note:
            'Goodness' here refers to a metric that indicates how well the model's current activations represent
            a given class. It's calculated by a function `layer_activations_to_goodness`, which transforms a
            layer's activations into a 'goodness' score. This function operates on the RecurrentFFNet model level
            and is called during the training process.
        """
        settings = Settings.new()

        for epoch in range(0, settings.model.epochs):
            logging.info("Epoch: " + str(epoch))

            for batch_num, (input_data, label_data) in enumerate(train_loader):
                input_data.move_to_device_inplace(self.settings.device.device)
                label_data.move_to_device_inplace(self.settings.device.device)

                if self.settings.model.should_replace_neg_data:
                    self.processor.replace_negative_data_inplace(
                        input_data.pos_input, label_data)

                average_layer_loss, pos_goodness_per_layer, neg_goodness_per_layer = self.__train_batch(
                    batch_num, input_data, label_data)

            # Get some observability into prediction while training.
            accuracy = self.processor.brute_force_predict(
                test_loader, 1)

            self.__log_metrics(accuracy, average_layer_loss,
                               pos_goodness_per_layer, neg_goodness_per_layer)

    def __train_batch(self, batch_num, input_data, label_data):
        logging.info("Batch: " + str(batch_num))

        self.inner_layers.reset_activations(True)

        # TODO: zero label
        for preinit_step in range(0, len(self.inner_layers)):
            logging.debug("Preinitialization step: " +
                          str(preinit_step))

            pos_input = input_data.pos_input[0]
            neg_input = input_data.neg_input[0]
            pos_labels = label_data.pos_labels[0]
            neg_labels = label_data.neg_labels[0]

            self.inner_layers.advance_layers_forward(
                ForwardMode.PositiveData, pos_input, pos_labels, False)
            self.inner_layers.advance_layers_forward(
                ForwardMode.NegativeData, neg_input, neg_labels, False)

        pos_goodness_per_layer = []
        neg_goodness_per_layer = []
        pos_target_latents = LatentAverager()
        iterations = input_data.pos_input.shape[0]
        for iteration in range(0, iterations):
            logging.debug("Iteration: " + str(iteration))

            input_data_sample = (
                input_data.pos_input[iteration],
                input_data.neg_input[iteration])
            label_data_sample = (
                label_data.pos_labels[iteration],
                label_data.neg_labels[iteration])

            total_loss = self.inner_layers.advance_layers_train(
                input_data_sample, label_data_sample, True)
            average_layer_loss = (total_loss / len(self.inner_layers)).item()
            logging.debug("Average layer loss: " +
                          str(average_layer_loss))

            if iteration >= self.data_config.focus_iteration_neg_offset and \
                    iteration <= self.data_config.focus_iteration_pos_offset:
                pos_goodness_per_layer.append([layer_activations_to_goodness(
                    layer.pos_activations.current).mean() for layer in self.inner_layers])
                neg_goodness_per_layer.append([layer_activations_to_goodness(
                    layer.neg_activations.current).mean() for layer in self.inner_layers])

                positive_latents = [
                    layer.pos_activations.current for layer in self.inner_layers]
                positive_latents = torch.cat(positive_latents, dim=1)
                pos_target_latents.track_collapsed_latents(positive_latents)

        if self.settings.model.should_replace_neg_data:
            pos_target_latents = pos_target_latents.retrieve()
            self.processor.train_class_predictor_from_latents(
                pos_target_latents, label_data.pos_labels[0])

        pos_goodness_per_layer = [
            sum(layer_goodnesses) /
            len(layer_goodnesses) for layer_goodnesses in zip(
                *
                pos_goodness_per_layer)]
        neg_goodness_per_layer = [
            sum(layer_goodnesses) /
            len(layer_goodnesses) for layer_goodnesses in zip(
                *
                neg_goodness_per_layer)]

        return average_layer_loss, pos_goodness_per_layer, neg_goodness_per_layer

    def __log_metrics(
            self,
            accuracy,
            average_layer_loss,
            pos_goodness_per_layer,
            neg_goodness_per_layer):
        # Supports wandb tracking of max 3 layer goodnesses
        try:
            first_layer_pos_goodness = pos_goodness_per_layer[0]
            first_layer_neg_goodness = neg_goodness_per_layer[0]
            second_layer_pos_goodness = pos_goodness_per_layer[1]
            second_layer_neg_goodness = neg_goodness_per_layer[1]
            third_layer_pos_goodness = pos_goodness_per_layer[2]
            third_layer_neg_goodness = neg_goodness_per_layer[2]
        except BaseException:
            # No-op as there may not be 3 layers
            pass

        if len(self.inner_layers) == 3:
            wandb.log({"acc": accuracy,
                       "loss": average_layer_loss,
                       "first_layer_pos_goodness": first_layer_pos_goodness,
                       "second_layer_pos_goodness": second_layer_pos_goodness,
                       "third_layer_pos_goodness": third_layer_pos_goodness,
                       "first_layer_neg_goodness": first_layer_neg_goodness,
                       "second_layer_neg_goodness": second_layer_neg_goodness,
                       "third_layer_neg_goodness": third_layer_neg_goodness})
        elif len(self.inner_layers) == 2:
            wandb.log({"acc": accuracy,
                       "loss": average_layer_loss,
                       "first_layer_pos_goodness": first_layer_pos_goodness,
                       "second_layer_pos_goodness": second_layer_pos_goodness,
                       "first_layer_neg_goodness": first_layer_neg_goodness,
                       "second_layer_neg_goodness": second_layer_neg_goodness})
        elif len(self.inner_layers) == 1:
            wandb.log({"acc": accuracy,
                       "loss": average_layer_loss,
                       "first_layer_pos_goodness": first_layer_pos_goodness,
                       "first_layer_neg_goodness": first_layer_neg_goodness})
