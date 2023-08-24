import logging

from torch import nn


class InnerLayers(nn.Module):

    def __init__(self, settings, ff_layers, conv_layers=None):
        super(InnerLayers, self).__init__()

        self.settings = settings

        self.ff_layers = ff_layers
        self.conv_layers = conv_layers

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

        if self.conv_layers is not None:
            input_data = self._process_convolutional_layers_multi_input(
                input_data)

        total_loss = 0
        for i, layer in enumerate(self.ff_layers):
            logging.debug("Training layer " + str(i))
            loss = None
            if i == 0 and len(self.ff_layers) == 1:
                loss = layer.train(input_data, label_data, should_damp)
            elif i == 0:
                loss = layer.train(input_data, None, should_damp)
            elif i == len(self.ff_layers) - 1:
                loss = layer.train(None, label_data, should_damp)
            else:
                loss = layer.train(None, None, should_damp)
            total_loss += loss
            logging.debug("Loss for layer " + str(i) + ": " + str(loss))

        logging.debug("Trained activations for layer " +
                      str(i))

        for layer in self.ff_layers:
            layer.advance_stored_activations()

        return total_loss

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
        if self.conv_layers is not None:
            input_data = self._process_convolutional_layers_single_input(
                input_data)

        for i, layer in enumerate(self.ff_layers):
            if i == 0 and len(self.ff_layers) == 1:
                layer.forward(mode, input_data, label_data, should_damp)
            elif i == 0:
                layer.forward(mode, input_data, None, should_damp)
            elif i == len(self.ff_layers) - 1:
                layer.forward(mode, None, label_data, should_damp)
            else:
                layer.forward(mode, None, None, should_damp)

        for layer in self.ff_layers:
            layer.advance_stored_activations()

    def reset_activations(self, isTraining):
        for layer in self.ff_layers:
            layer.reset_activations(isTraining)

    def _process_convolutional_layers_single_input(self, input_data):
        batch_size, flat_dim = input_data.shape
        square_image_width = int(flat_dim**0.5)

        reshaped_tensor = input_data.view(
            batch_size, square_image_width, square_image_width)
        reshaped_tensor = reshaped_tensor.unsqueeze(1)
        post_conv_input = self.conv_layers.forward(reshaped_tensor)
        post_conv_input = post_conv_input.contiguous(
        ).view(-1, post_conv_input.size(1) * post_conv_input.size(2) * post_conv_input.size(3))

        return post_conv_input

    def _process_convolutional_layers_multi_input(self, input_data):
        pos_input, neg_input = input_data

        pos_post_conv_input = self._process_convolutional_layers_single_input(
            pos_input)
        neg_post_conv_input = self._process_convolutional_layers_single_input(
            neg_input)

        return pos_post_conv_input, neg_post_conv_input

    def __len__(self):
        return len(self.ff_layers)

    def __iter__(self):
        return (layer for layer in self.ff_layers)
