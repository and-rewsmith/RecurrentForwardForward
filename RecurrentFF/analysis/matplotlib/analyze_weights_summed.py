import torch
import numpy as np
import matplotlib.pyplot as plt

LAYERS = 5


def plot_summed_weights(summed_weights, layer):
    """
    Plot the summed weights for a given layer to inspect potential cancellations.
    """
    plt.hist(summed_weights, bins=50, alpha=0.7)
    plt.title(f'Summed Weights Distribution - Layer {layer}')
    plt.xlabel('Summed Weight Value')
    plt.ylabel('Number of Neurons')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'img/weight_cancellations/Summed_Weights_Layer_{layer}.png')
    plt.close()


# Load the weights
weights = torch.load("IdentityMoneyball.pth", map_location=torch.device('cpu'))

# Compute and plot the summed weights for each layer
for layer in range(LAYERS):
    forward_key = f'inner_layers.layers.{layer}.forward_linear.weight'
    backward_key = f'inner_layers.layers.{layer}.backward_linear.weight'
    lateral_key = f'inner_layers.layers.{layer}.lateral_linear.parametrizations.weight.original'

    # Extract the weights for the current layer
    forward_weights = weights[forward_key].cpu().numpy()
    backward_weights = weights[backward_key].cpu().numpy()
    lateral_weights = weights[lateral_key].cpu().numpy()

    # Compute the summed weights for the current layer
    summed_weights = forward_weights.sum(
        axis=1) + backward_weights.sum(axis=1) + lateral_weights.sum(axis=1)

    # Plot the summed weights for the current layer
    plot_summed_weights(summed_weights, layer)
