from enum import Enum
import math
from typing import Dict, Optional, cast
from typing_extensions import Self

import wandb
import torch
from torchviz import make_dot
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import RMSprop, Adam, Adadelta, Optimizer, SGD
from torch.optim.lr_scheduler import StepLR
from profilehooks import profile  # type: ignore

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


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, block_size: int, bleed_factor: float = 0.0, bias: bool = False):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        # assert block_size * \
        #     2 < in_features, 'Block size must be less than half of the input features'
        # assert block_size * \
        #     2 < out_features, 'Block size must be less than half of the output features'

        if in_features == out_features:
            self.block_size_i = block_size
            self.block_size_j = block_size
        elif in_features > out_features:
            self.block_size_i = math.ceil(
                in_features / (out_features // block_size))
            self.block_size_j = block_size
        else:
            self.block_size_i = block_size
            self.block_size_j = math.ceil(
                out_features / (in_features // block_size))

        self.block_size = block_size
        self.bleed_factor = bleed_factor  # New parameter to control bleeding
        self.register_buffer('mask', self.create_mask())

    def create_mask(self):
        mask = torch.zeros(self.weight.size())

        # Compute how much additional overlap is allowed based on the bleed factor
        bleed_size_i = int(self.block_size_i * self.bleed_factor)
        bleed_size_j = int(self.block_size_j * self.bleed_factor)

        i = 0
        for j in range(0, self.out_features, self.block_size_j):
            # Set the mask for the block and the bleed regions
            mask[j:j+self.block_size_j+bleed_size_j,
                 i:i+self.block_size_i+bleed_size_i] = 1
            i = i + self.block_size_i

        # Clip the mask to the matrix size (in case of overflow due to bleeding)
        return mask[:self.out_features, :self.in_features]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # return F.linear(input, self.weight * self.mask, self.bias)
        return F.linear(input, self.weight, self.bias)

    def visualize_connectivity(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.weight.data * self.mask, cmap='viridis')
        plt.title(
            f'Connectivity Pattern (Block Size: {self.block_size}, Bleed Factor: {self.bleed_factor})')
        plt.colorbar()
        plt.show()


class ResidualConnection(nn.Module):
    """
    ResidualConnection class for inter-layer skip connections across HiddenLayers.
    """

    def __init__(self,
                 source: 'HiddenLayer',
                 target_size: int,
                 dropout_percentage: float,
                 initialization: WeightInitialization,
                 block_size: int,
                 bleed_factor: float) -> None:
        super(ResidualConnection, self).__init__()

        self.weight_initialization = initialization
        self.source = source
        self.weights = MaskedLinear(
            source.size, target_size, block_size=block_size, bleed_factor=bleed_factor)
        self.dropout = nn.Dropout(p=dropout_percentage)

        if initialization == WeightInitialization.Forward:
            nn.init.kaiming_uniform_(
                self.weights.weight, nonlinearity='relu')
        elif initialization == WeightInitialization.Backward:
            nn.init.uniform_(self.weights.weight, -0.05, 0.05)
        elif initialization == WeightInitialization.Lateral:
            raise AssertionError(
                "Lateral connections should not be initialized with this constructor")

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

        out: torch.Tensor = self.dropout(
            F.linear(source_activations_stdized, self.weights.weight, self.weights.bias))

        out = self.weights(source_activations_stdized)

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
            damping_factor: float,
            layer_num: int):
        super(HiddenLayer, self).__init__()

        self.size = size
        self.next_size = next_size
        self.prev_size = prev_size
        self.layer_num = layer_num

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

        self.generative_linear = nn.Sequential(
            # nn.Linear(size, size),
            # nn.ReLU(),
            # nn.Linear(size, size),
            # nn.ReLU(),
            nn.Linear(size, settings.data_config.data_size +
                      settings.data_config.num_classes)
        )
        for layer in self.generative_linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        connection_profile = self.settings.model.connection_profile
        self.forward_linear = MaskedLinear(
            prev_size, size, bleed_factor=connection_profile.forward_block_bleed[layer_num],
            block_size=connection_profile.forward_block_sizes[layer_num])
        nn.init.kaiming_uniform_(
            self.forward_linear.weight, nonlinearity='relu')

        self.forward_linear_inverse = MaskedLinear(
            size, prev_size, bleed_factor=connection_profile.forward_block_bleed[layer_num],
            block_size=connection_profile.forward_block_sizes[layer_num])
        nn.init.kaiming_uniform_(
            self.forward_linear_inverse.weight, nonlinearity='relu')

        if next_size == self.settings.data_config.num_classes:
            self.backward_linear = nn.Linear(next_size, size, bias=False)
            amplified_initialization(self.backward_linear, 3.0)
            self.backward_linear_inverse = nn.Linear(size, next_size, bias=False)
            nn.init.kaiming_uniform_(
                self.backward_linear_inverse.weight, nonlinearity='relu')
            # amplified_initialization(self.backward_linear_inverse, 1.0)
        else:
            self.backward_linear = MaskedLinear(
                next_size, size, bleed_factor=connection_profile.backward_block_bleed[layer_num],
                block_size=connection_profile.backward_block_sizes[layer_num])
            nn.init.uniform_(self.backward_linear.weight, -0.05, 0.05)
            self.backward_linear_inverse = MaskedLinear(
                size, next_size, bleed_factor=connection_profile.backward_block_bleed[layer_num],
                block_size=connection_profile.backward_block_sizes[layer_num])
            nn.init.uniform_(self.backward_linear_inverse.weight, -0.05, 0.05)

        # Initialize the lateral weights to be the identity matrix
        self.lateral_linear = MaskedLinear(
            size, size, block_size=connection_profile.lateral_block_sizes[layer_num],
            bleed_factor=connection_profile.lateral_block_bleed[layer_num])
        nn.init.orthogonal_(self.lateral_linear.weight, gain=math.sqrt(2))

        self.previous_layer: Self = None  # type: ignore[assignment]
        self.next_layer: Self = None  # type: ignore[assignment]

        self.forward_act: Tensor
        self.backward_act: Tensor
        self.lateral_act: Tensor

        self.inverse_optimizer = SGD(
            self.parameters(),
            lr=0.0001)
        self.inverse_criterion = nn.MSELoss()

        self.should_train = False

        self.reconstruction_losses = []

    def init_residual_connection(self, residual_connection: ResidualConnection) -> None:
        self.residual_connections.append(residual_connection)

    def filtered_named_parameters(self):
        """
        Returns an iterator over the named parameters of the current layer,
        excluding any parameters that start with 'previous_layer' or 'next_layer'.
        Each item in the iterator is a tuple of (name, param).
        """
        return ((name, param) for name, param in self.named_parameters()
                if not name.startswith('previous_layer') and not name.startswith('next_layer'))

    def filtered_parameters(self):
        """
        Returns an iterator over the parameters of the current layer,
        excluding any parameters that start with 'previous_layer'.
        """
        return (param for name, param in self.named_parameters()
                if not name.startswith('previous_layer') and not name.startswith('next_layer'))

    def filtered_parameters_gen_too(self):
        """
        Returns an iterator over the parameters of the current layer,
        excluding any parameters that start with 'previous_layer'.
        """
        return (param for name, param in self.named_parameters()
                if not name.startswith('previous_layer') and not name.startswith('next_layer') and not name.startswith('generative_linear'))

    def init_optimizer(self) -> None:
        self.optimizer: Optimizer
        if self.settings.model.ff_optimizer == "adam":
            self.optimizer = Adam(self.parameters(),
                                  lr=self.settings.model.ff_adam.learning_rate)
        elif self.settings.model.ff_optimizer == "rmsprop":
            self.optimizer = RMSprop(
                self.filtered_parameters(),
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
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            pos_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            self.pos_activations = Activations(
                pos_activations_current, pos_activations_previous)

            neg_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            neg_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            self.neg_activations = Activations(
                neg_activations_current, neg_activations_previous)

            self.predict_activations = None

        else:
            activations_dim = self.test_activations_dim

            predict_activations_current = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
                self.settings.device.device)
            predict_activations_previous = torch.zeros(
                activations_dim[0], activations_dim[1]).to(
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

    def reset_parameters_with_small_gradients(model: nn.Module, threshold: float = 1e-6):
        """
        Resets parameters that have gradients below a certain threshold to a standard initialization.
        Only considers parameters that don't start with 'previous_layer', 'next_layer', or 'generative_linear'.

        Args:
        model (nn.Module): The model whose parameters need to be checked and possibly reset.
        threshold (float): The gradient threshold below which parameters will be reset.
        """
        def parameter_filter(name_param_tuple):
            name, _ = name_param_tuple
            return not name.startswith('previous_layer') and not name.startswith('next_layer') and not name.startswith(
                'generative_linear')

        total_reset = 0
        with torch.no_grad():
            for name, param in filter(parameter_filter, model.named_parameters()):
                if param.grad is not None:
                    # Create a mask for parameters with small gradients
                    mask = torch.abs(param.grad.data) < threshold
                    num_reset = torch.sum(mask).item()
                    total_reset += num_reset

                    if num_reset > 0:
                        # Calculate fan_in (assuming the parameter is a weight matrix)
                        fan_in = param.size(1) if len(
                            param.size()) > 1 else param.size(0)

                        # Calculate the standard deviation for Kaiming initialization
                        std = math.sqrt(2.0 / fan_in)

                        # Generate new values for the parameters to be reset
                        new_values = torch.randn_like(param.data[mask]) * std

                        # Update only the parameters that need to be reset
                        param.data[mask] = new_values

                #         # print(f"Reset {num_reset} parameters in {name} (shape {param.shape}) due to small gradients")
                #     else:
                #         # print(f"No parameters reset in {name} (shape {param.shape})")
                # else:
                #     # print(f"No gradient for {name} (shape {param.shape})")

        wandb.log({"reset_parameters": total_reset})

    def generate_lpl_loss_predictive(
            self, current_activations_with_grad: torch.Tensor, prev_act: torch.Tensor) -> Tensor:
        def generate_loss(current_act: Tensor, previous_act: Tensor) -> Tensor:
            loss = torch.abs((current_act - previous_act))
            loss = torch.sum(loss, dim=1)
            loss = torch.sum(loss, dim=0)
            loss = loss / \
                (2 * current_act.shape[0] * current_act.shape[1])
            return loss

        assert current_activations_with_grad.requires_grad == True
        assert self.pos_activations.previous.requires_grad == False
        pos_loss = generate_loss(
            current_activations_with_grad, prev_act)
        return pos_loss

    # @profile(stdout=False, filename='baseline.prof',
    #          skip=Settings.new().model.skip_profiling)
    def train_layer(self,  # type: ignore[override]
                    input_data: TrainInputData,
                    label_data: TrainLabelData,
                    should_damp: bool,
                    retain_graph: bool) -> float:
        self.optimizer.zero_grad()

        pos_activations = None
        neg_activations = None
        if input_data is not None and label_data is not None:
            (pos_input, neg_input) = input_data
            (pos_labels, neg_labels) = label_data
            pos_activations, neg_activations = self.forward(
                ForwardMode.PositiveData, pos_input, pos_labels, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, neg_input, neg_labels, should_damp)
        elif input_data is not None:
            (pos_input, neg_input) = input_data
            pos_activations, neg_activations = self.forward(
                ForwardMode.PositiveData, pos_input, None, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, neg_input, None, should_damp)
        elif label_data is not None:
            (pos_labels, neg_labels) = label_data
            pos_activations, neg_activations = self.forward(
                ForwardMode.PositiveData, None, pos_labels, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, None, neg_labels, should_damp)
        else:
            pos_activations, neg_activations = self.forward(
                ForwardMode.PositiveData, None, None, should_damp)
            # neg_activations = self.forward(
            #     ForwardMode.NegativeData, None, None, should_damp)

        # smooth_loss_pos = self.generate_lpl_loss_predictive(
        #     pos_activations, self.pos_activations.previous)
        # smooth_loss_neg = self.generate_lpl_loss_predictive(
        #     neg_activations, self.neg_activations.previous)

        pos_badness = layer_activations_to_badness(pos_activations)
        neg_badness = layer_activations_to_badness(neg_activations)

        # if abs(pos_badness.mean() - neg_badness.mean()) < 0.1:
        #     self.should_train = True

        wandb.log({"pos_badness_loss": pos_badness.mean()}, step=self.settings.total_batch_count)
        wandb.log({"neg_badness_loss": neg_badness.mean()}, step=self.settings.total_batch_count)
        # print(neg_badness[0])
        # print(pos_badness[0])
        # print()
        # print(neg_badness_1[0:10])
        # print(neg_badness_2[0:10])
        # input()
        # print(neg_badness[0])
        # print(pos_badness[0])
        # input()
        # neg_badness = layer_activations_to_badness(neg_activations)

        alpha = 4
        delta = (pos_badness - neg_badness)
        # pos_badness = torch.clamp(pos_badness, min=0.5)
        # neg_badness = torch.clamp(neg_badness, max=3)

        # Loss function equivelent to:
        # plot3d log(1 + exp(-n + 1)) + log(1 + exp(p - 1)) for n=0 to 3, p=0
        # to 3
        contrastive_loss_0: Tensor = 0 * F.softplus(delta).mean()
        contrastive_loss_1: Tensor = 1 * F.softplus(torch.cat([
            (-1 * neg_badness) + self.settings.model.loss_threshold,
            pos_badness - self.settings.model.loss_threshold
        ])).mean()
        smooth_loss = torch.tensor(0)
        # smooth_loss = 0.0 * (smooth_loss_pos + smooth_loss_neg)
        # layer_loss = smooth_loss + contrastive_loss_0 + contrastive_loss_1
        layer_loss = contrastive_loss_0 + contrastive_loss_1
        # layer_loss = torch.clamp(layer_loss, max=20)
        layer_loss.backward(retain_graph=retain_graph)
        # layer_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # # go through all layers and collect their parameters
        # # print(self.named_parameters())
        # dot = make_dot(layer_loss, params=dict(self.named_parameters()))
        # dot.render('model_graph_outer', format='png')
        # input()

        # self.optimizer.step()

        # self.reset_parameters_with_small_gradients()
        # return cast(float, layer_loss.item()), contrastive_loss_0.item() + contrastive_loss_1.item(), smooth_loss.item()
        return cast(float, layer_loss.item()), layer_loss.item(), smooth_loss.item()

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
        # print("why?????")
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

            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)
            # prev_layer_stdized = prev_layer_prev_timestep_activations

            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)
            # next_layer_stdized = next_layer_prev_timestep_activations

            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)
            # prev_act_stdized = prev_act

            self.forward_act = self.forward_linear.forward(prev_layer_stdized)
            # print("0")
            # print(next_layer_stdized[0:10])
            self.backward_act = self.backward_linear.forward(next_layer_stdized)
            self.lateral_act = self.lateral_linear.forward(prev_act_stdized)

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
            # prev_act_stdized = prev_act

            self.forward_act = self.forward_linear.forward(data)
            # print("1")
            # print(labels[0:10])
            self.backward_act = self.backward_linear.forward(labels)
            self.lateral_act = self.lateral_linear.forward(prev_act_stdized)

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

            next_layer_stdized = standardize_layer_activations(
                next_layer_prev_timestep_activations, self.settings.model.epsilon)
            # next_layer_stdized = next_layer_prev_timestep_activations

            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)
            # prev_act_stdized = prev_act

            self.forward_act = self.forward_linear.forward(data)
            # print("2")
            # print(next_layer.pos_activations.previous[0:10])
            # print(next_layer.pos_activations.current[0:10])
            # print(next_layer_stdized[0:10])
            self.backward_act = self.backward_linear.forward(next_layer_stdized)
            self.lateral_act = self.lateral_linear.forward(prev_act_stdized)

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

            prev_layer_stdized = standardize_layer_activations(
                prev_layer_prev_timestep_activations, self.settings.model.epsilon)
            # prev_layer_stdized = prev_layer_prev_timestep_activations

            prev_act_stdized = standardize_layer_activations(
                prev_act, self.settings.model.epsilon)
            # prev_act_stdized = prev_act

            self.forward_act = self.forward_linear.forward(prev_layer_stdized)
            # print("3")
            # print(labels[0:10])
            self.backward_act = self.backward_linear.forward(labels)
            self.lateral_act = self.lateral_linear.forward(prev_act_stdized)

        # self.forward_act = self.forward_dropout(self.forward_act)
        # self.backward_act = self.backward_dropout(self.backward_act)
        # self.lateral_act = self.lateral_dropout(self.lateral_act)

        summation_act = self.forward_act + self.backward_act
        # summation_act = self.forward_act + self.backward_act

        # for residual_connection in self.residual_connections:
        #     summation_act = summation_act + residual_connection.forward(mode)

        new_activation = F.leaky_relu(summation_act)

        self.inverse_optimizer.zero_grad()

        # neg_input_forwards = F.linear(-self.backward_act.detach().clone(),
        #                               self.forward_linear.weight.T)
        neg_input_forwards = F.linear(self.backward_act.detach().clone(),
                                      self.forward_linear_inverse.weight)
        # inverse_loss.backward()
        # self.inverse_optimizer.step()
        neg_contribution_forwards = F.linear(
            neg_input_forwards,
            self.forward_linear.weight)
            # self.forward_linear.weight.detach().clone())
        inverse_loss_forwards = self.inverse_criterion(neg_contribution_forwards, self.forward_act.detach().clone())
        neg_contribution_forwards_redo = F.linear(
            neg_input_forwards.detach().clone(),
            self.forward_linear.weight)
        # neg_1_summation_act = neg_contribution_forwards_redo + self.backward_act.detach().clone()
        # new_activation_neg_1 = F.leaky_relu(neg_1_summation_act)

        # neg_input_backwards = F.linear(-self.forward_act.detach().clone(),
        #                                self.backward_linear.weight.T)
        neg_input_backwards = F.linear(self.forward_act.detach().clone(),
                                       self.backward_linear_inverse.weight)
        neg_contribution_backwards = F.linear(
            neg_input_backwards,
            self.backward_linear.weight)
        inverse_loss_backwards = self.inverse_criterion(neg_contribution_backwards, self.backward_act.detach().clone())
        neg_contribution_backwards_redo = F.linear(
            neg_input_backwards.detach().clone(),
            self.backward_linear.weight)
        # neg_2_summation_act = neg_contribution_backwards_redo + self.forward_act.detach().clone()
        # new_activation_neg_2 = F.leaky_relu(neg_2_summation_act)
        new_activation_neg = F.leaky_relu(neg_contribution_forwards_redo + neg_contribution_backwards_redo)

        # if self.layer_num == 0 and self.should_train:
        #     # print("neg_1_summation_act norm:")
        #     # print(torch.norm(neg_1_summation_act, p=2))
        #     wandb.log({"neg_1_summation_act_norm": torch.norm(neg_1_summation_act, p=2)},
        #               step=self.settings.total_batch_count)

        #     print()

        if inverse_loss_backwards.requires_grad:
            total_loss = inverse_loss_backwards + inverse_loss_forwards
            self.reconstruction_losses.append(total_loss.item())
            # if not self.should_train:
            total_loss.backward(retain_graph=True)
            self.inverse_optimizer.step()
            self.inverse_optimizer.zero_grad()

        # print(self.backward_linear.weight.requires_grad)
        # print(new_activation_neg_1.grad)
        # print(new_activation_neg_2.grad)
        # print(None)

        # self.analyze_activations(self.forward_act, self.backward_act, self.forward_linear, self.backward_linear)

        # print("summations:---------")
        # print(neg_1_summation_act[0:10])
        # print(neg_2_summation_act[0:10])
        # print("activations (fw / bw):---------")
        # print(self.forward_act[0:10])
        # print(self.backward_act[0:10])
        # print()
        # input()

        if should_damp:
            old_activation = new_activation
            new_activation = (1 - self.damping_factor) * \
                prev_act + self.damping_factor * old_activation
            new_activation_neg = (1 - self.damping_factor) * \
                prev_act + self.damping_factor * new_activation_neg

        if mode == ForwardMode.PositiveData:
            assert self.pos_activations is not None
            self.pos_activations.current = new_activation
            # self.pos_activations.current.requires_grad = False
        elif mode == ForwardMode.NegativeData:
            assert self.neg_activations is not None
            self.neg_activations.current = new_activation
            # self.neg_activations.current.requires_grad = False
        elif mode == ForwardMode.PredictData:
            assert self.predict_activations is not None
            self.predict_activations.current = new_activation
            # self.predict_activations.current.requires_grad = False

        # print(new_activation.shape)
        # print(new_activation_neg_1.shape)
        # print(new_activation_neg_2.shape)
        # input()

        return new_activation, new_activation_neg
