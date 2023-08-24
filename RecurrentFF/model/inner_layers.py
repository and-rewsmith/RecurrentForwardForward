import logging


from torch import nn
import wandb


class InnerLayers(nn.Module):

    def __init__(self, settings, layers):
        super(InnerLayers, self).__init__()

        self.settings = settings

        self.layers = layers

    def advance_layers_train(self, input_data, label_data, should_damp):
        """
        Advances the training process for all layers in the network by computing
        the loss for each layer and updating their activations.

        The method handles different layer scenarios: if it's a single layer,
        both the input data and label data are used for training. If it's the
        first or last layer in a multi-layer configuration, only the input data
        or label data is used, respectively. For layers in the middle of a
        multi-layer network, neither the input data nor the label data is used.

        Args:
            input_data (torch.Tensor): The input data for the network.

            label_data (torch.Tensor): The target labels for the network.

            should_damp (bool): A flag to determine whether the activation
            damping should be applied during training.

        Returns:
            total_loss (float): The accumulated loss over all layers in the
            network during the current training step.

        Note:
            The layer's 'train' method is expected to return a loss value, which
            is accumulated to compute the total loss for the network. After
            training each layer, their stored activations are advanced by
            calling the 'advance_stored_activations' method.
        """
        losses_per_layer = [0 for _ in self.layers]
        for i, layer in enumerate(self.layers):
            logging.debug("Training layer " + str(i))
            loss = None
            if i == 0 and len(self.layers) == 1:
                loss = layer.train(input_data, label_data, should_damp)
            elif i == 0:
                loss = layer.train(input_data, None, should_damp)
            elif i == len(self.layers) - 1:
                loss = layer.train(None, label_data, should_damp)
            else:
                loss = layer.train(None, None, should_damp)

            layer_num = i+1
            logging.debug("Loss for layer " +
                          str(layer_num) + ": " + str(loss))

            metric_name = "granular_loss (layer " + str(layer_num) + ")"
            wandb.log({metric_name: loss})

            losses_per_layer[i] += loss

        logging.debug("Trained activations for layer " +
                      str(i))

        for layer in self.layers:
            layer.advance_stored_activations()

        return losses_per_layer

    def advance_layers_forward(
            self,
            mode,
            input_data,
            label_data,
            should_damp):
        """
        Executes a forward pass through all layers of the network using the
        given mode, input data, label data, and a damping flag.

        The method handles different layer scenarios: if it's a single layer,
        both the input data and label data are used for the forward pass. If
        it's the first or last layer in a multi-layer configuration, only the
        input data or label data is used, respectively. For layers in the middle
        of a multi-layer network, neither the input data nor the label data is
        used.

        After the forward pass, the method advances the stored activations for
        all layers.

        Args:
            mode (ForwardMode): An enum representing the mode of forward
            propagation. This could be PositiveData, NegativeData, or
            PredictData.

            input_data (torch.Tensor): The input data for the
            network.

            label_data (torch.Tensor): The target labels for the
            network.

            should_damp (bool): A flag to determine whether the
            activation damping should be applied during the forward pass.

        Note:
            This method doesn't return any value. It modifies the internal state
            of the layers by performing a forward pass and advancing their
            stored activations.
        """
        for i, layer in enumerate(self.layers):
            if i == 0 and len(self.layers) == 1:
                layer.forward(mode, input_data, label_data, should_damp)
            elif i == 0:
                layer.forward(mode, input_data, None, should_damp)
            elif i == len(self.layers) - 1:
                layer.forward(mode, None, label_data, should_damp)
            else:
                layer.forward(mode, None, None, should_damp)

        for layer in self.layers:
            layer.advance_stored_activations()

    def reset_activations(self, isTraining):
        for layer in self.layers:
            layer.reset_activations(isTraining)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return (layer for layer in self.layers)
