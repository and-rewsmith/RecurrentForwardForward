import logging
import math


from torch import nn
import torch
import wandb

from RecurrentFF.model.hidden_layer import HiddenLayer


class InnerLayers(nn.Module):

    def __init__(self, settings, layers):
        super(InnerLayers, self).__init__()

        self.settings = settings

        self.layers = layers

    def advance_layers_train(self, input_data, label_data, should_damp, layer_metrics):
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

            layer_metrics.ingest_layer_metrics(i, layer, loss)

        layer_metrics.increment_samples_seen()

        logging.debug("Trained activations for layer " +
                      str(i))

        for layer in self.layers:
            layer.advance_stored_activations()

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


class LayerMetrics:
    def __init__(self, num_layers: int):
        self.pos_activations_norms = [0 for _ in range(0, num_layers)]
        self.neg_activations_norms = [0 for _ in range(0, num_layers)]
        self.forward_weights_norms = [0 for _ in range(0, num_layers)]
        self.forward_grads_norms = [0 for _ in range(0, num_layers)]
        self.backward_weights_norms = [0 for _ in range(0, num_layers)]
        self.backward_grads_norms = [0 for _ in range(0, num_layers)]
        self.lateral_weights_norms = [0 for _ in range(0, num_layers)]
        self.lateral_grads_norms = [0 for _ in range(0, num_layers)]
        self.losses_per_layer = [0 for _ in range(0, num_layers)]

        self.num_data_points = 0

        self.update_norms = {}
        self.momentum_norms = {}
        self.update_angles = {}

    def ingest_layer_metrics(self, layer_num: int, layer: HiddenLayer, loss: int):
        pos_activations_norm = torch.norm(layer.pos_activations.current, p=2)
        neg_activations_norm = torch.norm(layer.neg_activations.current, p=2)
        forward_weights_norm = torch.norm(layer.forward_linear.weight, p=2)
        backward_weights_norm = torch.norm(layer.backward_linear.weight, p=2)
        lateral_weights_norm = torch.norm(layer.lateral_linear.weight, p=2)

        forward_grad_norm = torch.norm(layer.forward_linear.weight.grad, p=2)
        backward_grads_norm = torch.norm(
            layer.backward_linear.weight.grad, p=2)
        lateral_grads_norm = torch.norm(layer.lateral_linear.weight.grad, p=2)

        # forward_grad_norm = torch.norm(
        #     layer.forward_linear.parametrizations.weight.original.grad, p=2)
        # backward_grads_norm = torch.norm(
        #     layer.backward_linear.parametrizations.weight.original.grad, p=2)
        # lateral_grads_norm = torch.norm(
        #     layer.lateral_linear.parametrizations.weight.original.grad, p=2)

        self.pos_activations_norms[layer_num] += pos_activations_norm
        self.neg_activations_norms[layer_num] += neg_activations_norm
        self.forward_weights_norms[layer_num] += forward_weights_norm
        self.forward_grads_norms[layer_num] += forward_grad_norm
        self.backward_weights_norms[layer_num] += backward_weights_norm
        self.backward_grads_norms[layer_num] += backward_grads_norm
        self.lateral_weights_norms[layer_num] += lateral_weights_norm
        self.lateral_grads_norms[layer_num] += lateral_grads_norm
        self.losses_per_layer[layer_num] += loss

        for group in layer.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # compute update norm
                    update = -group['lr'] * param.grad
                    update_norm = torch.norm(update, p=2)
                    param_name = layer.param_name_dict[param]

                    if layer_num not in self.update_norms:
                        self.update_norms[layer_num] = {}
                    if param_name not in self.update_norms[layer_num]:
                        self.update_norms[layer_num][param_name] = 0

                    self.update_norms[layer_num][param_name] += update_norm

                    # compute momentum norm
                    try:
                        momentum_norm = torch.norm(
                            layer.optimizer.state[param]['momentum_buffer'])
                        if layer_num not in self.momentum_norms:
                            self.momentum_norms[layer_num] = {}
                        if param_name not in self.momentum_norms[layer_num]:
                            self.momentum_norms[layer_num][param_name] = 0

                        self.momentum_norms[layer_num][param_name] += momentum_norm
                    except (KeyError, AttributeError):
                        logging.debug(
                            "No momentum buffer for param. Assume using non-momentum optimizer.")

                    # compute angle
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        param.grad.view(-1), update.view(-1), dim=0)
                    cosine_similarity = torch.clamp(cosine_similarity, -1, 1)
                    angle_in_degrees = torch.acos(
                        cosine_similarity) * (180 / math.pi)
                    if layer_num not in self.update_angles:
                        self.update_angles[layer_num] = {}
                    if param_name not in self.update_angles[layer_num]:
                        self.update_angles[layer_num][param_name] = 0

                    self.update_angles[layer_num][param_name] += angle_in_degrees

    def increment_samples_seen(self):
        self.num_data_points += 1

    def average_layer_loss(self):
        return sum(self.losses_per_layer) / self.num_data_points

    def log_metrics(self, total_batch_count):
        for i in range(0, len(self.pos_activations_norms)):
            layer_num = i+1

            metric_name = "pos_activations_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.pos_activations_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "neg_activations_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.neg_activations_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "forward_weights_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.forward_weights_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "forward_grad_norms (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.forward_grads_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "backward_weights_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.backward_weights_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "backward_grad_norms (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.backward_grads_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "lateral_weights_norms (layer " + \
                str(layer_num) + ")"
            wandb.log(
                {metric_name: self.lateral_weights_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "lateral_grad_norms (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.lateral_grads_norms[i] / self.num_data_points}, step=total_batch_count)

            metric_name = "loss (layer " + str(layer_num) + ")"
            wandb.log(
                {metric_name: self.losses_per_layer[i] / self.num_data_points}, step=total_batch_count)

        for layer_index in self.update_norms:
            layer_index_display = layer_index + 1
            for param_name in self.update_norms[layer_index]:
                metric_name = f"{param_name} update norm (layer {str(layer_index_display)})"
                wandb.log(
                    {metric_name: self.update_norms[layer_index]
                        [param_name] / self.num_data_points},
                    step=total_batch_count)

        for layer_index in self.momentum_norms:
            layer_index_display = layer_index + 1
            for param_name in self.momentum_norms[layer_index]:
                metric_name = f"{param_name} momentum (layer {str(layer_index_display)})"
                wandb.log(
                    {metric_name: self.momentum_norms[layer_index]
                        [param_name] / self.num_data_points},
                    step=total_batch_count)

        for layer_index in self.update_angles:
            layer_index_display = layer_index + 1
            for param_name in self.update_angles[layer_index]:
                metric_name = f"{param_name} update angle (layer {str(layer_index_display)})"
                wandb.log(
                    {metric_name: self.update_angles[layer_index]
                        [param_name] / self.num_data_points},
                    step=total_batch_count)
