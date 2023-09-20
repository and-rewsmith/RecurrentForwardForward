import numpy as np
import matplotlib.pyplot as plt
import torch

LAYERS = 5


def plot_heatmap(weights, layer_idx, vmin, vmax):
    """Plot the heatmap of weights for a specific layer."""

    forward_key = f"inner_layers.layers.{layer_idx}.forward_linear.weight"
    recurrent_key = f"inner_layers.layers.{layer_idx}.lateral_linear.parametrizations.weight.original"
    backward_key = f"inner_layers.layers.{layer_idx}.backward_linear.weight"

    # Extract weights
    forward_weights = weights[forward_key].detach().cpu().numpy()
    recurrent_weights = weights[recurrent_key].detach().cpu().numpy()
    backward_weights = weights[backward_key].detach().cpu().numpy()

    weight_types = {
        "Forward": forward_weights,
        "Recurrent": recurrent_weights,
        "Backward": backward_weights
    }

    for name, weight_matrix in weight_types.items():
        plt.figure(figsize=(8, 8))
        plt.imshow(weight_matrix, cmap='viridis',
                   vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar()
        plt.title(f'Layer {layer_idx} - {name} Weights')
        plt.xlabel('Neurons')
        plt.ylabel('Neurons')
        plt.tight_layout()
        plt.savefig(
            f"img/weight_heatmaps/layer_{layer_idx}_{name}_heatmap.png")


def get_global_min_max(weights, layers):
    """Find the global minimum and maximum values across all weights."""
    all_weights = []
    for i in range(layers):
        forward_key = f"inner_layers.layers.{i}.forward_linear.weight"
        recurrent_key = f"inner_layers.layers.{i}.lateral_linear.parametrizations.weight.original"
        backward_key = f"inner_layers.layers.{i}.backward_linear.weight"

        all_weights.extend([
            weights[forward_key].detach().cpu().numpy().flatten(),
            weights[recurrent_key].detach().cpu().numpy().flatten(),
            weights[backward_key].detach().cpu().numpy().flatten()
        ])

    all_weights = np.concatenate(all_weights)
    return np.min(all_weights), np.max(all_weights)


def plot_density_function(weights, layer_idx):
    """Plot the density function of weight sparsities for a specific layer."""

    forward_key = f"inner_layers.layers.{layer_idx}.forward_linear.weight"
    recurrent_key = f"inner_layers.layers.{layer_idx}.lateral_linear.parametrizations.weight.original"
    backward_key = f"inner_layers.layers.{layer_idx}.backward_linear.weight"

    # Extract and flatten weights
    forward_weights = weights[forward_key].detach().cpu().numpy().flatten()
    recurrent_weights = weights[recurrent_key].detach().cpu().numpy().flatten()
    backward_weights = weights[backward_key].detach().cpu().numpy().flatten()

    weight_types = {
        "forward": forward_weights,
        "recurrent": recurrent_weights,
        "backward": backward_weights
    }

    plt.figure(figsize=(10, 6))

    # Create a set of threshold values based on the percentiles of weights
    thresholds = np.linspace(
        0, np.max([np.max(w) for w in weight_types.values()]), 100)

    for name, weight_values in weight_types.items():
        densities = [np.mean(weight_values < t) for t in thresholds]
        plt.plot(thresholds, densities, label=name)

    plt.xlabel('Thresholds')
    plt.ylabel('Percentage of weights below threshold')
    plt.title(f'Layer {layer_idx} Density Function of Weight Sparsities')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(
        f"img/weight_sparsities/layer_{layer_idx}_density_function.png")


# Assuming you're using the same loading code as before
weights = torch.load("IdentityMoneyball.pth", map_location=torch.device('cpu'))

# for i in range(layers):
#     plot_density_function(weights, i)


# Get global min and max values for consistent scale
vmin, vmax = get_global_min_max(weights, LAYERS)
scaling_factor = 6
vmin = vmin // scaling_factor
vmax = vmax // scaling_factor

# Plot the heatmaps
for i in range(LAYERS):
    plot_heatmap(weights, i, vmin, vmax)
