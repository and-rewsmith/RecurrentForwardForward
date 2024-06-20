from dataclasses import make_dataclass
from enum import Enum
import math
import random
from typing import Dict, Optional, cast
from typing_extensions import Self

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import RMSprop, Adam, Adadelta, Optimizer
from torch.optim.lr_scheduler import StepLR
from profilehooks import profile
from torchviz import make_dot


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


class WeightInitialization(Enum):
    Forward = 1
    Backward = 2
    Lateral = 3


class ResidualConnection(nn.Module):
    """
    ResidualConnection class for inter-layer skip connections across HiddenLayers.
    """

    def __init__(self,
                 source: 'HiddenLayer',
                 target_size: int,
                 dropout_percentage: float,
                 initialization: WeightInitialization):
        super(ResidualConnection, self).__init__()

        self.weight_initialization = initialization
        self.source = source
        self.weights = nn.Linear(source.size, target_size)
        self.dropout = nn.Dropout(p=dropout_percentage)

        if initialization == WeightInitialization.Forward:
            nn.init.kaiming_uniform_(
                self.weights.weight, nonlinearity='relu')
        elif initialization == WeightInitialization.Backward:
            nn.init.uniform_(self.weights.weight, -0.05, 0.05)
        elif initialization == WeightInitialization.Lateral:
            raise AssertionError("Lateral connections should not be initialized with this constructor")

    def train(self: Self, mode: bool = True) -> Self:
        self.training = mode
        return self

    def eval(self: Self) -> Self:
        return self.train(False)

    def forward(self, mode: ForwardMode) -> torch.Tensor:
        if mode == ForwardMode.PositiveData:
            assert self.source.pos_activations is not None
            source_activations = self.source.pos_activations.previous
        elif mode == ForwardMode.NegativeData:
            assert self.source.neg_activations is not None
            source_activations = self.source.neg_activations.previous
        elif mode == ForwardMode.PredictData:
            assert self.source.predict_activations is not None
            source_activations = self.source.predict_activations.previous

        source_activations_stdized = standardize_layer_activations(
            source_activations, self.source.settings.model.epsilon)

        out: torch.Tensor = self.dropout(F.linear(source_activations_stdized, self.weights.weight, self.weights.bias))

        if self.weight_initialization == WeightInitialization.Backward:
            out = -1 * out

        return out

    def _apply(self, fn):  # type: ignore
        """
        Override apply, but we don't want to apply to sibling layers as that
        will cause a stack overflow. The hidden layers are contained in a
        collection in the higher-level RecurrentFFNet. They will all get the
        apply call from there.
        """
        # Remove 'source' temporarily
        source = self.source
        self.source = None

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

        # Restore 'source'
        self.source = source

        return self

    def state_dict(self, *args, **kwargs):  # type: ignore
        # Remove 'source' temporarily
        source = self.source
        self.source = None

        # Get the state dict without the linked layers
        state = super().state_dict(*args, **kwargs)

        # Restore 'source'
        self.source = source
        return state


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

        self.size = size
        self.next_size = next_size
        self.prev_size = prev_size

        self.residual_connections = nn.ModuleList()

        self.settings = settings

        self.train_activations_dim = (train_batch_size, size)
        self.test_activations_dim = (test_batch_size, size)

        self.damping_factor = damping_factor

        self.pos_activations: Optional[Activations] = None
        self.neg_activations: Optional[Activations] = None
        self.predict_activations: Optional[Activations] = None
        self.reset_activations(True)

        self.forward_dropout = nn.Dropout(p=self.settings.model.dropout)
        self.backward_dropout = nn.Dropout(p=self.settings.model.dropout)
        self.lateral_dropout = nn.Dropout(p=self.settings.model.dropout)

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

        self.forward_act: Tensor
        self.backward_act: Tensor
        self.lateral_act: Tensor

    def init_residual_connection(self, residual_connection: ResidualConnection) -> None:
        self.residual_connections.append(residual_connection)

    def init_optimizer(self) -> None:
        self.optimizer: Optimizer
        # param_optimizer = [(name, param) for name, param in self.named_parameters()]
        # for (name, param) in param_optimizer:
        #     print(name)
        #     print(param.shape)
        # input()
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

    def train(self: Self, mode: bool = True) -> Self:
        self.training = mode
        return self

    def eval(self: Self) -> Self:
        return self.train(False)

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

            pos_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1], requires_grad=False).to(
                self.settings.device.device)
            pos_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1], requires_grad=False).to(
                self.settings.device.device)
            self.pos_activations = Activations(
                pos_activations_current, pos_activations_previous)

            neg_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1], requires_grad=False).to(
                self.settings.device.device)
            neg_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1], requires_grad=False).to(
                self.settings.device.device)
            self.neg_activations = Activations(
                neg_activations_current, neg_activations_previous)

            self.predict_activations = None

        else:
            activations_dim = self.test_activations_dim

            predict_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1], requires_grad=False).to(
                self.settings.device.device)
            predict_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1], requires_grad=False).to(
                self.settings.device.device)
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

    def generate_lpl_loss_predictive(self, current_activations_with_grad: torch.Tensor) -> Tensor:
        def generate_loss(current_act: Tensor, previous_act: Tensor) -> Tensor:
            loss = (current_act - previous_act) ** 2
            loss = torch.sum(loss, dim=1)
            loss = torch.sum(loss, dim=0)
            loss = loss / \
                (2 * current_act.shape[0] * current_act.shape[1])
            return loss

        assert current_activations_with_grad.requires_grad == True
        assert self.pos_activations.previous.requires_grad == False
        pos_loss = generate_loss(
            current_activations_with_grad, self.pos_activations.previous)
        return pos_loss

    def generate_lpl_loss_hebbian(self, current_activations_with_grad: torch.Tensor) -> Tensor:
        def generate_loss(activations: Tensor) -> Tensor:
            mean_act = torch.mean(activations, dim=0)
            # print("mean_act:", mean_act)
            mean_subtracted = activations - mean_act
            # print("mean: ", mean_subtracted)

            sigma_squared = torch.sum(
                mean_subtracted ** 2, dim=0) / (activations.shape[0] - 1)
            # print("sigma: ", sigma_squared)

            loss = -torch.log(sigma_squared + 0.00000001).sum() / sigma_squared.shape[0]
            # print("loss: ", loss)
            return loss

        assert current_activations_with_grad.requires_grad == True
        pos_loss = generate_loss(current_activations_with_grad)
        return pos_loss

    def generate_lpl_loss_decorrelative(self, current_activations_with_grad: torch.Tensor) -> Tensor:
        def generate_loss(activations: torch.Tensor) -> torch.Tensor:
            # Compute the mean across the batch dimension
            mean_act = torch.mean(activations, dim=0)
            # print("mean_act", mean_act)

            # Subtract mean from activations and square the result
            deviations = (activations - mean_act) ** 2
            # print("deviations", deviations[0])

            # Outer product along feature dimension for each batch
            # This computes the pairwise squared differences efficiently
            loss = torch.einsum('bi,bj->bij', deviations, deviations)
            # print("loss", loss[0])

            # Sum over all batches and features, exclude the diagonal elements
            # Diagonal elements correspond to the squared terms which we want to avoid
            batch_size, n_features = activations.shape
            loss = torch.sum(loss) - \
                torch.sum(torch.einsum('bii->b', loss)) / 2
            # print("loss", loss)

            # Normalize the loss
            # print("batch size * n_features * (n_features - 1)", batch_size * n_features * (n_features - 1))
            loss = loss / (batch_size * n_features * (n_features - 1))
            # print("loss", loss)

            return loss

        assert current_activations_with_grad.requires_grad == True
        pos_loss = generate_loss(current_activations_with_grad)
        return pos_loss

    # @profile(stdout=False, filename='baseline.prof',
    #          skip=Settings.new().model.skip_profiling)

    def train_layer(self,  # type: ignore[override]
                    input_data: TrainInputData,
                    label_data: TrainLabelData,
                    should_damp: bool) -> float:
        self.optimizer.zero_grad()

        pos_activations = None
        # neg_activations = None
        if input_data is not None and label_data is not None:
            try:
                (pos_input, neg_input) = input_data
                (pos_labels, neg_labels) = label_data
            except ValueError:
                pos_input = input_data
                pos_labels = label_data

            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, pos_labels, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, neg_input, neg_labels, should_damp)
        elif input_data is not None:
            try:
                (pos_input, neg_input) = input_data
            except ValueError:
                pos_input = input_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, None, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, neg_input, None, should_damp)
        elif label_data is not None:
            try:
                (pos_labels, neg_labels) = label_data
            except ValueError:
                pos_labels = label_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, None, pos_labels, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, None, neg_labels, should_damp)
        else:
            pos_activations = self.forward(
                ForwardMode.PositiveData, None, None, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, None, None, should_damp)

        pos_badness = layer_activations_to_badness(pos_activations)
        ff_layer_loss: Tensor = F.softplus(
            pos_badness - self.settings.model.loss_threshold
        ).mean()
        ff_layer_loss = self.settings.model.loss_scale_ff * ff_layer_loss

        # y = 4exp(-log(x+1.5))
        ff_layer_loss_min: Tensor = torch.sqrt(torch.square(pos_activations) + 0.001).mean()
        ff_layer_loss_min = 4 * torch.exp(-torch.log(ff_layer_loss_min + self.settings.model.loss_threshold))

        # lpl_loss_predictive: Tensor = self.settings.model.loss_scale_predictive * \
        #     self.generate_lpl_loss_predictive(pos_activations)
        # lpl_loss_hebbian: Tensor = self.settings.model.loss_scale_hebbian * \
        #     self.generate_lpl_loss_hebbian(pos_activations)
        # lpl_loss_decorrelative: Tensor = self.settings.model.loss_scale_decorrelative * \
        #     self.generate_lpl_loss_decorrelative(pos_activations)

        assert ff_layer_loss.requires_grad == True
        assert ff_layer_loss_min.requires_grad == True
        # assert lpl_loss_predictive.requires_grad == True
        # assert lpl_loss_hebbian.requires_grad == True
        # assert lpl_loss_decorrelative.requires_grad == True

        # if random.random() < 0.005:
        #     # print("pos_act: ", pos_activations)
        #     print("ff_layer_loss: ", ff_layer_loss)
        #     print("lpl_loss_predictive: ", lpl_loss_predictive)
        #     print("lpl_loss_hebbian: ", lpl_loss_hebbian)
        #     print("lpl_loss_decorrelative: ", lpl_loss_decorrelative)
        #     print()

        # layer_loss: Tensor = ff_layer_loss + lpl_loss_predictive + \
        #     lpl_loss_hebbian + lpl_loss_decorrelative + ff_layer_loss_min
        layer_loss: Tensor = ff_layer_loss + ff_layer_loss_min

        # print(ff_layer_loss)
        # print(lpl_loss_predictive)
        # print(lpl_loss_hebbian)
        # print(lpl_loss_decorrelative)

        # important block
        #
        # print("ff_layer_loss: ", ff_layer_loss)
        # print("lpl_loss_predictive: ", lpl_loss_predictive)
        # print("lpl_loss_hebbian: ", lpl_loss_hebbian)
        # print("lpl_loss_decorrelative: ", lpl_loss_decorrelative)
        # print(self.size)

        # print(pos_activations[0])
        # input()

        # print(layer_loss)
        # for name, param in self.named_parameters():
        #     print(name, param.grad)
        # optimizer_params = []
        # for d in self.optimizer.param_groups:
        #     for tensor in d['params']:
        #         print(tensor.shape)
        #         optimizer_params.append(tensor)
        # print("optimizer params")
        # input()
        # print()
        layer_loss.backward()

        # params = dict(self.named_parameters())
        # dot = make_dot(layer_loss, params=params)
        # dot.render('computation_graph', format='png')
        # input()

        # print all grads of parameters
        # for name, param in self.named_parameters():
        #     print(name, param.grad)
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

            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations
            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)

            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations
            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                prev_layer_stdized,
                self.forward_linear.weight,
                self.forward_linear.bias)
            self.backward_act = -1 * F.linear(
                next_layer_stdized,
                self.backward_linear.weight,
                self.backward_linear.bias)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight,
                self.lateral_linear.bias)

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

            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                data,
                self.forward_linear.weight,
                self.forward_linear.bias)
            self.backward_act = -1 * F.linear(
                labels,
                self.backward_linear.weight,
                self.backward_linear.bias)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight,
                self.lateral_linear.bias)

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

            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations
            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                data,
                self.forward_linear.weight,
                self.forward_linear.bias)
            self.backward_act = -1 * F.linear(
                next_layer_stdized,
                self.backward_linear.weight,
                self.backward_linear.bias)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight,
                self.lateral_linear.bias)

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

            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations
            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)

            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)

            self.forward_act = F.linear(
                prev_layer_stdized,
                self.forward_linear.weight,
                self.forward_linear.bias)
            self.backward_act = -1 * F.linear(
                labels,
                self.backward_linear.weight,
                self.backward_linear.bias)
            self.lateral_act = F.linear(
                prev_act_stdized,
                self.lateral_linear.weight,
                self.lateral_linear.bias)

        self.forward_act = self.forward_dropout(self.forward_act)
        self.backward_act = self.backward_dropout(self.backward_act)
        self.lateral_act = self.lateral_dropout(self.lateral_act)

        summation_act = self.forward_act + self.backward_act + self.lateral_act

        for residual_connection in self.residual_connections:
            summation_act = summation_act + residual_connection.forward(mode)

        new_activation = F.leaky_relu(summation_act)

        if should_damp:
            old_activation = new_activation
            new_activation = (1 - self.damping_factor) * \
                prev_act + self.damping_factor * old_activation

        if mode == ForwardMode.PositiveData:
            assert self.pos_activations is not None
            self.pos_activations.current = new_activation.detach()
        elif mode == ForwardMode.NegativeData:
            assert self.neg_activations is not None
            self.neg_activations.current = new_activation.detach()
        elif mode == ForwardMode.PredictData:
            assert self.predict_activations is not None
            self.predict_activations.current = new_activation.detach()

        return new_activation
