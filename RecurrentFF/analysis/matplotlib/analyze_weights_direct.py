import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(correlation_matrix, layer):
    """
    Plot the correlation heatmap for a given layer.
    """
    sns.heatmap(correlation_matrix, annot=True,
                cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title(f'Weight Correlations - Layer {layer}')
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=[
               'Forward', 'Backward', 'Lateral'])
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=[
               'Forward', 'Backward', 'Lateral'])
    plt.tight_layout()
    plt.savefig(f'img/weight_cancellations/Correlation_Layer_{layer}.png')
    plt.close()


def plot_weights_comparison(forward_weights, backward_weights, lateral_weights, neuron_idx, layer):
    """
    Plot the weights comparison for a specific neuron in a given layer.
    """
    plt.hist(forward_weights[neuron_idx], bins=50,
             alpha=0.5, label='Forward', density=True)
    plt.hist(backward_weights[neuron_idx], bins=50,
             alpha=0.5, label='Backward', density=True)
    plt.hist(lateral_weights[neuron_idx], bins=50,
             alpha=0.5, label='Lateral', density=True)
    plt.title(f'Weights Comparison - Neuron {neuron_idx} in Layer {layer}')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(
        f'img/weight_cancellations/Comparison_Neuron_{neuron_idx}_Layer_{layer}.png')
    plt.close()


# Load the weights
weights = torch.load('weights_9-2.pth')

# Number of layers in the network
layers = len(weights) // 6

# Perform the analyses for each layer
for layer in range(layers):
    forward_key = f'inner_layers.layers.{layer}.forward_linear.weight'
    backward_key = f'inner_layers.layers.{layer}.backward_linear.weight'
    lateral_key = f'inner_layers.layers.{layer}.lateral_linear.weight'

    # Extract the weights for the current layer
    forward_weights_layer = weights[forward_key].cpu().numpy()
    backward_weights_layer = weights[backward_key].cpu().numpy()
    lateral_weights_layer = weights[lateral_key].cpu().numpy()

    # For Correlation Analysis
    forward_weights = forward_weights_layer.flatten()
    backward_weights = backward_weights_layer.flatten()
    lateral_weights = lateral_weights_layer.flatten()

    # Find the minimum length among the flattened weights
    min_length = min(len(forward_weights), len(
        backward_weights), len(lateral_weights))

    # Truncate weights to the minimum length
    forward_weights = forward_weights[:min_length]
    backward_weights = backward_weights[:min_length]
    lateral_weights = lateral_weights[:min_length]

    # Compute the correlation between the truncated weights
    combined_weights = np.stack(
        [forward_weights, backward_weights, lateral_weights])
    correlation_matrix = np.corrcoef(combined_weights)

    # Plot the correlation heatmap for the current layer
    plot_correlation_heatmap(correlation_matrix, layer)

    # For Direct Comparison
    # Plot the weights comparison for a sample neuron (e.g., neuron 0) in the current layer
    plot_weights_comparison(
        forward_weights_layer, backward_weights_layer, lateral_weights_layer, 0, layer)
