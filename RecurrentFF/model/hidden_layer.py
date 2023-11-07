import math
from typing import Dict, Optional, cast
from typing_extensions import Self

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import RMSprop, Adam, Adadelta, Optimizer
from torch.optim.lr_scheduler import StepLR

from RecurrentFF.util import (
    Activations,
    ForwardMode,
    TrainInputData,
    TrainLabelData,
    layer_activations_to_badness,
    standardize_layer_activations,
)
from RecurrentFF.settings import (
    Settings,
)


def custom_load_state_dict(self, state_dict: Dict, strict=True):  # type: ignore
    # This function is a replication of the original PyTorch load_state_dict logic
    # with a check to prevent infinite recursion through the linked layers.
    def load(module: nn.Module, prefix=''):  # type: ignore
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs)
        for name, child in module._modules.items():
            # Check to prevent infinite recursion
            if name not in ['previous_layer', 'next_layer']:
                if child is not None:
                    load(child, prefix + name + '.')

    missing_keys = []  # type: ignore
    unexpected_keys = []  # type: ignore
    error_msgs = []  # type: ignore

    # The original function uses _IncompatibleKeys to track this, but for simplicity
    # we'll just use two lists and construct it at the end if needed.

    metadata = getattr(state_dict, '_metadata', None)
    load(self)

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError(
            'Error(s) in loading state_dict:\n\t{}'.format(
                '\n\t'.join(error_msgs)))

    return self


def amplified_initialization(layer: nn.Linear, amplification_factor: float = 3.0) -> None:
    """Amplified initialization for Linear layers."""
    # Get the number of input features
    n = layer.in_features
    # Compute the standard deviation for He initialization
    std = (2.0 / n) ** 0.5
    # Amplify the standard deviation
    amplified_std = std * amplification_factor
    # Initialize weights with amplified standard deviation
    nn.init.normal_(layer.weight, mean=0, std=amplified_std)


class HiddenLayer(nn.Module):
    """
    A HiddenLayer class for a novel Forward-Forward Recurrent Network, with
    inspiration drawn from Boltzmann Machines and Noise Contrastive Estimation.
    This network design is characterized by two distinct forward passes, each
    with specific objectives: one is dedicated to processing positive ("real")
    data with the aim of lowering the 'badness' across every hidden layer,
    while the other is tasked with processing negative data and adjusting the
    weights to increase the 'badness' metric.

    The HiddenLayer is essentially a node within this network, with possible
    connections to both preceding and succeeding layers, depending on its
    specific location within the network architecture. The first layer in this
    setup is connected directly to the input data, and the last layer maintains
    a connection to the output data. The intermediate layers establish a link to
    both their previous and next layers, if available.

    In each HiddenLayer, a forward linear transformation and a backward linear
    transformation are defined. The forward transformation is applied to the
    activations from the previous layer, while the backward transformation is
    applied to the activations of the subsequent layer. The forward
    transformation helps in propagating the data through the network, and the
    backward transformation is key in the learning process where it aids in the
    adjustment of weights based on the output or next layer's activations.
    """

    setattr(Module, "load_state_dict", custom_load_state_dict)

    def __init__(
            self,
            settings: Settings,
            train_batch_size: int,
            test_batch_size: int,
            prev_size: int,
            size: int,
            next_size: int,
            damping_factor: float):
        super(HiddenLayer, self).__init__()

        self.settings = settings

        self.train_activations_dim = (train_batch_size, size)
        self.test_activations_dim = (test_batch_size, size)

        self.damping_factor = damping_factor

        self.pos_activations: Optional[Activations] = None
        self.neg_activations: Optional[Activations] = None
        self.predict_activations: Optional[Activations] = None

        self.stable_state_activations: Optional[torch.Tensor] = None

        self.reset_activations(True)

        self.forward_linear = nn.Linear(prev_size, size)
        nn.init.kaiming_uniform_(
            self.forward_linear.weight, nonlinearity='relu')

        self.backward_linear = nn.Linear(next_size, size)

        if next_size == self.settings.data_config.num_classes:
            amplified_initialization(self.backward_linear, 3.0)
        else:
            nn.init.uniform_(self.backward_linear.weight, -0.05, 0.05)

        # Initialize the lateral weights to be the identity matrix
        self.lateral_linear = nn.Linear(size, size)
        nn.init.orthogonal_(self.lateral_linear.weight, gain=math.sqrt(2))

        self.previous_layer: Self = None  # type: ignore[assignment]
        self.next_layer: Self = None  # type: ignore[assignment]

        self.optimizer: Optimizer
        if self.settings.model.ff_optimizer == "adam":
            self.optimizer = Adam(self.parameters(),
                                  lr=self.settings.model.ff_adam.learning_rate)
        elif self.settings.model.ff_optimizer == "rmsprop":
            self.optimizer = RMSprop(
                self.parameters(),
                lr=self.settings.model.ff_rmsprop.learning_rate,
                momentum=self.settings.model.ff_rmsprop.momentum)
        elif self.settings.model.ff_optimizer == "adadelta":
            self.optimizer = Adadelta(
                self.parameters(),
                lr=self.settings.model.ff_adadelta.learning_rate)

        self.scheduler = StepLR(
            self.optimizer, step_size=self.settings.model.lr_step_size, gamma=self.settings.model.lr_gamma)

        self.param_name_dict = {param: name for name,
                                param in self.named_parameters()}

        self.forward_act: Tensor
        self.backward_act: Tensor
        self.lateral_act: Tensor

    def _apply(self, fn):  # type: ignore
        """
        Override apply, but we don't want to apply to sibling layers as that
        will cause a stack overflow. The hidden layers are contained in a
        collection in the higher-level RecurrentFFNet. They will all get the
        apply call from there.
        """
        # Remove `previous_layer` and `next_layer` temporarily
        previous_layer = self.previous_layer
        next_layer = self.next_layer
        self.previous_layer = None
        self.next_layer = None

        # Apply `fn` to each parameter and buffer of this layer
        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        # Apply `fn` to submodules
        for module in self.children():
            module._apply(fn)

        # Restore `previous_layer` and `next_layer`
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        return self

    def state_dict(self, *args, **kwargs):  # type: ignore
        # Temporarily unlink the previous and next layers
        previous_layer = self.previous_layer
        next_layer = self.next_layer
        self.previous_layer = None
        self.next_layer = None

        # Get the state dict without the linked layers
        state = super().state_dict(*args, **kwargs)

        # Restore the links
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        return state

    def step_learning_rate(self) -> None:
        self.scheduler.step()

    def reset_activations(self, isTraining: bool) -> None:
        activations_dim = None
        if isTraining:
            activations_dim = self.train_activations_dim

            if self.stable_state_activations is None:
                pos_activations_current = torch.zeros(
                    activations_dim[0], activations_dim[1]).to(
                    self.settings.device.device)
                pos_activations_previous = torch.zeros(
                    activations_dim[0], activations_dim[1]).to(
                    self.settings.device.device)
                neg_activations_current = torch.zeros(
                    activations_dim[0], activations_dim[1]).to(
                    self.settings.device.device)
                neg_activations_previous = torch.zeros(
                    activations_dim[0], activations_dim[1]).to(
                    self.settings.device.device)

            else:
                pos_activations_stable_state = self.stable_state_activations.retrieve_random_stable_state_activations(
                    self.settings.data_config.train_batch_size)
                neg_activations_stable_state = self.stable_state_activations.retrieve_random_stable_state_activations(
                    self.settings.data_config.train_batch_size)
                pos_activations_current = pos_activations_stable_state.clone()
                pos_activations_previous = pos_activations_stable_state.clone()
                neg_activations_current = neg_activations_stable_state.clone()
                neg_activations_previous = neg_activations_stable_state.clone()

            self.pos_activations = Activations(
                pos_activations_current, pos_activations_previous)
            self.neg_activations = Activations(
                neg_activations_current, neg_activations_previous)
            self.predict_activations = None

        else:
            activations_dim = self.test_activations_dim

            if self.stable_state_activations is None:
                predict_activations_current = torch.zeros(
                    activations_dim[0], activations_dim[1]).to(
                    self.settings.device.device)
                predict_activations_previous = torch.zeros(
                    activations_dim[0], activations_dim[1]).to(
                    self.settings.device.device)
            else:
                activations_stable_state = self.stable_state_activations.retrieve_random_stable_state_activations(
                    self.settings.data_config.test_batch_size)
                predict_activations_current = activations_stable_state.clone()
                predict_activations_previous = activations_stable_state.clone()

            self.predict_activations = Activations(
                predict_activations_current, predict_activations_previous)
            self.pos_activations = None
            self.neg_activations = None

    def advance_stored_activations(self) -> None:
        if self.pos_activations is not None:
            self.pos_activations.advance()

        if self.neg_activations is not None:
            self.neg_activations.advance()

        if self.predict_activations is not None:
            self.predict_activations.advance()

    def set_previous_layer(self, previous_layer: Self) -> None:
        self.previous_layer = previous_layer

    def set_next_layer(self, next_layer: Self) -> None:
        self.next_layer = next_layer

    def train(self,  # type: ignore[override]
              input_data: TrainInputData,
              label_data: TrainLabelData,
              should_damp: bool) -> float:
        self.optimizer.zero_grad()

        pos_activations = None
        neg_activations = None
        if input_data is not None and label_data is not None:
            (pos_input, neg_input) = input_data
            (pos_labels, neg_labels) = label_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, pos_labels, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, neg_input, neg_labels, should_damp)
        elif input_data is not None:
            (pos_input, neg_input) = input_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, None, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, neg_input, None, should_damp)
        elif label_data is not None:
            (pos_labels, neg_labels) = label_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, None, pos_labels, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, None, neg_labels, should_damp)
        else:
            pos_activations = self.forward(
                ForwardMode.PositiveData, None, None, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, None, None, should_damp)

        pos_badness = layer_activations_to_badness(pos_activations)
        neg_badness = layer_activations_to_badness(neg_activations)

        # Loss function equivelent to:
        # plot3d log(1 + exp(-n + 1)) + log(1 + exp(p - 1)) for n=0 to 3, p=0
        # to 3
        layer_loss: Tensor = F.softplus(torch.cat([
            (-1 * neg_badness) + self.settings.model.loss_threshold,
            pos_badness - self.settings.model.loss_threshold
        ])).mean()
        layer_loss.backward()

        self.optimizer.step()
        return cast(float, layer_loss.item())

    # TODO: needs to be more DRY
    def forward(self, mode: ForwardMode, data: torch.Tensor, labels: torch.Tensor, should_damp: bool) -> torch.Tensor:
        """
        Propagates input data forward through the network, updating the
        activation state of the current layer based on the operating mode.

        Handles various scenarios depending on the layer configuration in the
        network (input layer, output layer, or a middle layer).

        Args:
            mode (ForwardMode enum): Indicates the type of data being propagated
            (positive, negative, or prediction).

            data (torch.Tensor or None): The input data for the layer. If
            `None`, it indicates that this layer is not the input layer.

            labels (torch.Tensor or None): The target labels for the layer. If
            `None`, it indicates that this layer is not the output layer.

            should_damp (bool): A flag to determine whether the activation
            damping should be applied.

        Returns:
            new_activation (torch.Tensor): The updated activation state of the
            layer after the forward propagation.

        Note:
            'Damping' here refers to a technique used to smoothen the changes in
            the layer activations over time. In this function, damping is
            implemented as a weighted average of the previous and the newly
            computed activations, controlled by the `self.damping_factor`.

            The function expects to receive input data and/or labels depending
            on the layer. The absence of both implies the current layer is a
            'middle' layer. If only the labels are missing, this layer is an
            'input' layer, while if only the data is missing, it's an 'output'
            layer. If both are provided, the network has only a single layer.

            All four scenarios are handled separately in the function, although
            the general procedure is similar: compute new activations based on
            the received inputs (and possibly, depending on the layer's
            position, the activations of the adjacent layers), optionally apply
            damping, update the current layer's activations, and return the new
            activations.
        """
        next_layer = self.next_layer
        previous_layer = self.previous_layer

        # Make sure assumptions aren't violated regarding layer connectivity.
        if data is None:
            assert previous_layer is not None
        if labels is None:
            assert next_layer is not None

        # Middle layer.
        new_activation: Tensor
        prev_act: Tensor = None  # type: ignore[assignment]
        if data is None and labels is None:
            next_layer_prev_timestep_activations = None
            prev_layer_prev_timestep_activations = None
            if mode == ForwardMode.PositiveData:
                next_layer_prev_timestep_activations = cast(
                    Activations, next_layer.pos_activations).previous
                prev_layer_prev_timestep_activations = cast(
                    Activations, previous_layer.pos_activations).previous
                prev_act = cast(Activations, self.pos_activations).previous
            elif mode == ForwardMode.NegativeData:
                next_layer_prev_timestep_activations = cast(
                    Activations, next_layer.neg_activations).previous
                prev_layer_prev_timestep_activations = cast(
                    Activations, previous_layer.neg_activations).previous
                prev_act = cast(Activations, self.neg_activations).previous
            elif mode == ForwardMode.PredictData:
                next_layer_prev_timestep_activations = cast(
                    Activations, next_layer.predict_activations).previous
                prev_layer_prev_timestep_activations = cast(
                    Activations, previous_layer.predict_activations).previous
                prev_act = cast(Activations, self.predict_activations).previous

            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations.detach()
            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)

            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations.detach()
            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                prev_layer_stdized,
                self.forward_linear.weight)
            self.backward_act = -1 * F.linear(
                next_layer_stdized,
                self.backward_linear.weight)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight)

        # Single layer scenario. Hidden layer connected to input layer and
        # output layer.
        elif data is not None and labels is not None:
            if mode == ForwardMode.PositiveData:
                assert self.pos_activations is not None
                prev_act = cast(Activations, self.pos_activations).previous
            elif mode == ForwardMode.NegativeData:
                assert self.neg_activations is not None
                prev_act = cast(Activations, self.neg_activations).previous
            elif mode == ForwardMode.PredictData:
                assert self.predict_activations is not None
                prev_act = cast(Activations, self.predict_activations).previous

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                data,
                self.forward_linear.weight)
            self.backward_act = -1 * F.linear(
                labels,
                self.backward_linear.weight)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight)

        # Input layer scenario. Connected to input layer and hidden layer.
        elif data is not None:
            next_layer_prev_timestep_activations = None
            if mode == ForwardMode.PositiveData:
                next_layer_prev_timestep_activations = cast(
                    Activations, next_layer.pos_activations).previous
                prev_act = cast(Activations, self.pos_activations).previous
            elif mode == ForwardMode.NegativeData:
                next_layer_prev_timestep_activations = cast(
                    Activations, next_layer.neg_activations).previous
                prev_act = cast(Activations, self.neg_activations).previous
            elif mode == ForwardMode.PredictData:
                next_layer_prev_timestep_activations = cast(
                    Activations, next_layer.predict_activations).previous
                prev_act = cast(Activations, self.predict_activations).previous

            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations.detach()
            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                data,
                self.forward_linear.weight)
            self.backward_act = -1 * F.linear(
                next_layer_stdized,
                self.backward_linear.weight)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight)

        # Output layer scenario. Connected to hidden layer and output layer.
        elif labels is not None:
            prev_layer_prev_timestep_activations = None
            if mode == ForwardMode.PositiveData:
                prev_layer_prev_timestep_activations = cast(
                    Activations, previous_layer.pos_activations).previous
                prev_act = cast(Activations, self.pos_activations).previous
            elif mode == ForwardMode.NegativeData:
                prev_layer_prev_timestep_activations = cast(
                    Activations, previous_layer.neg_activations).previous
                prev_act = cast(Activations, self.neg_activations).previous
            elif mode == ForwardMode.PredictData:
                prev_layer_prev_timestep_activations = cast(
                    Activations, previous_layer.predict_activations).previous
                prev_act = cast(Activations, self.predict_activations).previous

            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations.detach()
            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act = prev_act.detach()
            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                prev_layer_stdized,
                self.forward_linear.weight)
            self.backward_act = -1 * F.linear(
                labels,
                self.backward_linear.weight)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight)

        new_activation = F.leaky_relu(
            self.forward_act + self.backward_act + self.lateral_act)

        if should_damp:
            old_activation = new_activation
            new_activation = (1 - self.damping_factor) * \
                prev_act + self.damping_factor * old_activation

        if mode == ForwardMode.PositiveData:
            assert self.pos_activations is not None
            self.pos_activations.current = new_activation
        elif mode == ForwardMode.NegativeData:
            assert self.neg_activations is not None
            self.neg_activations.current = new_activation
        elif mode == ForwardMode.PredictData:
            assert self.predict_activations is not None
            self.predict_activations.current = new_activation

        return new_activation
