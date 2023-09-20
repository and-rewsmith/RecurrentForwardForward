import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_weight_distributions(weights_list, labels, title):
    for weights, label in zip(weights_list, labels):
        plt.hist(weights.flatten(), bins=100,
                 alpha=0.5, label=label, density=True)
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'img/weight_distributions/{title}.png')
    plt.close()


# Load the weights
weights = torch.load('weights_9-2.pth')

# Number of layers in the network
layers = len(weights) // 6

# Plot the distributions for each layer
for layer in range(layers):
    forward_key = f'inner_layers.layers.{layer}.forward_linear.weight'
    backward_key = f'inner_layers.layers.{layer}.backward_linear.weight'
    lateral_key = f'inner_layers.layers.{layer}.lateral_linear.weight'

    plot_weight_distributions(
        [weights[forward_key].cpu().numpy(), weights[backward_key].cpu().numpy(),
         weights[lateral_key].cpu().numpy()],
        ['Forward', 'Backward', 'Lateral'],
        f'Weight Distributions - Layer {layer}'
    )
