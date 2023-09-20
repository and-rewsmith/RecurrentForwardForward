import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F


def plot_activation_parts_breakdown():
    num_files = 10  # 0 through 9

    # Create subplots for each file
    fig, axes = plt.subplots(num_files, figsize=(14, 4 * num_files))

    # Loop to load and process each file
    for i in range(num_files):
        # Load the tensor
        data = torch.load(f'activations_{i}.pt')

        # Extract tensors and compute their means
        forward_mean = data['forward'].view(
            data['forward'].shape[0], -1).mean(dim=1).cpu().numpy()
        backward_mean = data['backward'].view(
            data['backward'].shape[0], -1).mean(dim=1).cpu().numpy()
        lateral_mean = data['lateral'].view(
            data['lateral'].shape[0], -1).mean(dim=1).cpu().numpy()

        # Compute sum for each timestep
        total_mean = forward_mean + backward_mean + lateral_mean

        # Plot for current tensor on its subplot
        ax = axes[i]
        ax.plot(forward_mean, label=f'Forward', alpha=0.5)
        ax.plot(backward_mean, label=f'Backward', alpha=0.5)
        ax.plot(lateral_mean, label=f'Lateral', alpha=0.5)
        ax.plot(total_mean, label=f'Sum', color='black', linewidth=2)

        ax.set_title(f'Tensor values for activations_{i}.pt')
        ax.legend()
        ax.grid(True)

    # Finalizing the plots
    plt.tight_layout()
    plt.savefig("img/activation_cancellations/0.png")


def plot_activation_heatmap(file_number):
    # Load the tensor
    data = torch.load(f'activations_{file_number}.pt')

    # Create figure and axes for the 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 25))

    activations = ['forward', 'backward', 'lateral']

    # Plotting the heatmaps for the 3 activations
    for idx, activation in enumerate(activations):
        activation_data = data[activation].squeeze(dim=1).cpu().numpy()
        ax = axes[idx]
        cax = ax.imshow(activation_data, aspect='auto',
                        cmap='viridis', interpolation='nearest', origin='lower', vmin=-50, vmax=30)
        fig.colorbar(cax, ax=ax)
        ax.set_title(
            f'{activation.capitalize()} Activations for activations_{file_number}.pt')
        ax.set_xlabel('Neuronal activations')
        ax.set_ylabel('Timesteps')

    # Compute the summed activations
    summed_activations = data['forward'] + data['backward'] + data['lateral']

    # Compute the summed activations with leaky_relu
    summed_activations_leaky = F.leaky_relu(
        summed_activations).squeeze(dim=1).cpu().numpy()

    summed_activations = summed_activations.squeeze(dim=1).cpu().numpy()

    # Plotting the heatmap for the summed activations without leaky_relu
    ax = axes[3]
    cax = ax.imshow(summed_activations, aspect='auto',
                    cmap='viridis', interpolation='nearest', origin='lower', vmin=-50, vmax=30)
    fig.colorbar(cax, ax=ax)
    ax.set_title(
        f'Summed Activations (No Leaky ReLU) for activations_{file_number}.pt')
    ax.set_xlabel('Neuronal activations')
    ax.set_ylabel('Timesteps')

    # Plotting the heatmap for the summed activations with leaky_relu
    ax = axes[4]
    cax = ax.imshow(summed_activations_leaky, aspect='auto',
                    cmap='viridis', interpolation='nearest', origin='lower', vmin=0, vmax=10)
    fig.colorbar(cax, ax=ax)
    ax.set_title(
        f'Summed Activations (With Leaky ReLU) for activations_{file_number}.pt')
    ax.set_xlabel('Neuronal activations')
    ax.set_ylabel('Timesteps')

    # Finalizing the plots
    plt.tight_layout()
    plt.savefig(f"img/activation_cancellations/heatmap_{file_number}.png")


if __name__ == '__main__':
    # For demo purposes, I'll call the heatmap function for file number 0.
    # You can adjust this or loop over multiple file numbers as needed.
    plot_activation_heatmap(0)
