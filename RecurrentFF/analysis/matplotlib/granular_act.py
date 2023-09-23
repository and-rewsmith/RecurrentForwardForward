import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


OUTPUT_ACTIVATION_LIMIT_LOWER = -0.5
OUTPUT_ACTIVATION_LIMIT_UPPER = 8

BASE_PT_PATH = "./artifacts/activations"

LAYERS = 5


def plot_cosine_similarity_multi_file(file_names, activation_type="correct"):
    # Initialize the activation accumulator
    accum_data = None

    for file_name in file_names:
        # Load the tensor for each file
        data = torch.load(f"{BASE_PT_PATH}/{file_name}")
        if accum_data is None:
            # If this is the first file, initialize accum_data with the same
            # keys and shapes as the loaded data
            accum_data = {key: torch.zeros_like(
                value.float()) for key, value in data.items()}

        # Accumulate activations from each file
        for key in accum_data.keys():
            accum_data[key] += data[key].float()

    # Compute the average
    n_files = len(file_names)
    for key in accum_data.keys():
        accum_data[key] /= n_files

    # Activation pair names
    basic_comparisons = [
        ('forward', 'backward'),
        ('forward', 'lateral'),
        ('backward', 'lateral')
    ]

    complex_comparisons = [
        ('forward', 'backward + lateral'),
        ('backward', 'forward + lateral'),
        ('lateral', 'forward + backward')
    ]

    # Number of layers
    n_layers = data["incorrect_lateral_activations"].shape[1]

    # Create figure and axes for the cosine similarity plots for each layer
    fig, axes = plt.subplots(n_layers, 2, figsize=(20, 20))

    for layer in range(n_layers):
        for col, comparisons in enumerate(
                [basic_comparisons, complex_comparisons]):
            ax = axes[layer, col]

            for act1, act2 in comparisons:
                # Fetch the data for the first activation type for the current
                # layer
                data1 = accum_data[f'{activation_type}_{act1}_activations'][:, layer, :].cpu(
                ).numpy()

                # For combined pairs like "backward + lateral", we add the
                # activations
                if '+' in act2:
                    act2_parts = act2.split('+')
                    act2_parts = [a.strip() for a in act2_parts]
                    data2 = sum(accum_data[f'{activation_type}_{a}_activations'][:, layer, :].cpu(
                    ).numpy() for a in act2_parts)
                else:
                    data2 = accum_data[f'{activation_type}_{act2}_activations'][:, layer, :].cpu(
                    ).numpy()

                all_data = np.concatenate([data1, data2], axis=0)
                pca = PCA(n_components=5).fit(all_data)
                data1_projected = pca.transform(data1)
                data2_projected = pca.transform(data2)

                # Compute cosine similarity for each time step
                cos_sim = [cosine_similarity(data1_projected[i].reshape(
                    1, -1), data2_projected[i].reshape(1, -1))[0][0] for i in range(data1_projected.shape[0])]

                ax.plot(cos_sim, label=f'Cosine Similarity: {act1} :: {act2}')
                ax.set_title(
                    f'Layer {layer + 1} - {"Basic" if col == 0 else "Complex"} Comparisons')
                ax.set_xlabel('Timesteps')
                ax.set_ylabel('Cosine Similarity')
                ax.legend()
                ax.set_ylim(-1, 1)  # Cosine similarity range

    # Finalizing the plots
    plt.tight_layout()
    plt.savefig(
        f"img/debug/activation_cancellations/cos_similarity_{activation_type}_average.png")


def plot_cosine_similarity(file_name, activation_type="correct"):
    # Load the tensor
    data = torch.load(f"{BASE_PT_PATH}/{file_name}")

    # Activation pair names
    basic_comparisons = [
        ('forward', 'backward'),
        ('forward', 'lateral'),
        ('backward', 'lateral')
    ]

    complex_comparisons = [
        ('forward', 'backward + lateral'),
        ('backward', 'forward + lateral'),
        ('lateral', 'forward + backward')
    ]

    # Number of layers
    n_layers = data["incorrect_lateral_activations"].shape[1]

    # Create figure and axes for the cosine similarity plots for each layer
    fig, axes = plt.subplots(n_layers, 2, figsize=(20, 20))

    for layer in range(n_layers):
        for col, comparisons in enumerate(
                [basic_comparisons, complex_comparisons]):
            ax = axes[layer, col]

            for act1, act2 in comparisons:
                # Fetch the data for the first activation type for the current
                # layer
                data1 = data[f'{activation_type}_{act1}_activations'][:, layer, :].cpu(
                )

                # For combined pairs like "backward + lateral", we add the
                # activations
                if '+' in act2:
                    act2_parts = act2.split('+')
                    act2_parts = [a.strip() for a in act2_parts]
                    data2 = sum(
                        data[f'{activation_type}_{a}_activations'][:, layer, :].cpu() for a in act2_parts)
                else:
                    data2 = data[f'{activation_type}_{act2}_activations'][:, layer, :].cpu(
                    )

                all_data = np.concatenate([data1, data2], axis=0)
                pca = PCA(n_components=5).fit(all_data)
                data1_projected = pca.transform(data1)
                data2_projected = pca.transform(data2)

                # Compute cosine similarity for each time step
                cos_sim = [
                    cosine_similarity(
                        data1_projected[i].reshape(1, -1),
                        data2_projected[i].reshape(1, -1))[0][0] for i in range(
                        data1.size(0))]

                ax.plot(cos_sim, label=f'Cosine Similarity: {act1} :: {act2}')
                ax.set_title(
                    f'Layer {layer + 1} - {"Basic" if col == 0 else "Complex"} Comparisons')
                ax.set_xlabel('Timesteps')
                ax.set_ylabel('Cosine Similarity')
                ax.legend()
                ax.set_ylim(-1, 1)  # Cosine similarity range

    # Finalizing the plots
    plt.tight_layout()
    plt.savefig(
        f"img/debug/activation_cancellations/cos_similarity_{activation_type}_{file_name.replace('.pt', '')}.png")


def plot_l2_norm_across_time(file_name, activation_type="correct"):
    # Load the tensor
    data = torch.load(f"{BASE_PT_PATH}/{file_name}")
    num_layers = data["incorrect_lateral_activations"].shape[1]

    # Create a figure and axes for the 3 rows (layers) and 2 columns (L2 norm
    # and mean) of subplots
    fig, axes = plt.subplots(num_layers, 2, figsize=(20, 15))

    activations_names = ['forward', 'backward', 'lateral']

    # Compute and plot the L2 norm and mean of the activations for the 3 layers
    for layer in range(num_layers):

        # Plotting L2 Norm
        ax_l2 = axes[layer, 0]
        sum_activations = np.zeros(
            data[f'{activation_type}_forward_activations'][:, layer, :].shape)

        for activation in activations_names:
            activation_key = f'{activation_type}_{activation}_activations'
            activation_data = data[activation_key][:, layer, :].cpu().numpy()

            # Compute L2 norm across time
            l2_norm = np.linalg.norm(activation_data, axis=1)
            ax_l2.plot(l2_norm, label=f"{activation.capitalize()} Activation")

            sum_activations += activation_data

        # # Plot the L2 norm of the summed activations
        # l2_norm_sum = np.linalg.norm(sum_activations, axis=1)
        # ax_l2.plot(l2_norm_sum, label="Summed Activations", linestyle="--")

        # Plot the L2 norm of the summed activations after applying leaky_relu
        leaky_relu_sum_activations = F.leaky_relu(
            torch.tensor(sum_activations)).numpy()
        l2_norm_leaky = np.linalg.norm(leaky_relu_sum_activations, axis=1)
        ax_l2.plot(l2_norm_leaky,
                   label="Leaky ReLU Summed Activations", linestyle="-.")
        ax_l2.axhline(0, color='gray', linestyle='--')
        ax_l2.set_title(
            f'L2 Norm of Activations for Layer {layer+1} in {file_name}')
        ax_l2.set_xlabel('Timesteps')
        ax_l2.set_ylabel('L2 Norm')
        ax_l2.legend()

        # Plotting Mean of Activations
        ax_mean = axes[layer, 1]
        sum_activations_mean = np.zeros(
            data[f'{activation_type}_forward_activations'][:, layer, :].shape)

        for activation in activations_names:
            activation_key = f'{activation_type}_{activation}_activations'
            activation_data = data[activation_key][:, layer, :].cpu().numpy()

            # Compute mean across time
            mean_activation = np.mean(activation_data, axis=1)
            ax_mean.plot(mean_activation,
                         label=f"{activation.capitalize()} Activation")

            sum_activations_mean += activation_data

        # # Plot the mean of the summed activations
        # mean_sum_activation = np.mean(sum_activations_mean, axis=1)
        # ax_mean.plot(mean_sum_activation,
        #              label="Summed Activations", linestyle="--")

        # Plot the mean of the summed activations after applying leaky_relu
        mean_leaky_sum_activation = np.mean(leaky_relu_sum_activations, axis=1)
        ax_mean.plot(mean_leaky_sum_activation,
                     label="Leaky ReLU Summed Activations", linestyle="-.")
        ax_mean.axhline(0, color='gray', linestyle='--')
        ax_mean.set_title(
            f'Mean of Activations for Layer {layer+1} in {file_name}')
        ax_mean.set_xlabel('Timesteps')
        ax_mean.set_ylabel('Mean Activation')
        ax_mean.legend()

    # Finalizing the plots
    plt.tight_layout()
    plt.savefig(
        f"img/debug/activation_cancellations/l2_and_mean_{activation_type}_{file_name.replace('.pt', '')}.png")


def plot_activation_heatmap(file_name, activation_type="correct"):
    # Load the tensor
    data = torch.load(f"{BASE_PT_PATH}/{file_name}")

    # Create figure and axes for the 6 subplots for n layers (1 additional
    # subplot for the combined activations)
    fig, axes = plt.subplots(6, LAYERS, figsize=(30, 30))

    activations_names = ['forward', 'backward', 'lateral']

    # Plotting the heatmaps for the 3 activations
    for idx, activation in enumerate(activations_names):
        for layer in range(LAYERS):
            activation_key = f'{activation_type}_{activation}_activations'
            activation_data = data[activation_key][:, layer, :].cpu().numpy()
            ax = axes[idx, layer]

            vmin = -50
            vmax = 30
            if "lateral" in activation_key.lower():
                vmin = -10
                vmax = 10

            cax = ax.imshow(
                activation_data,
                aspect='auto',
                cmap='viridis',
                interpolation='nearest',
                origin='lower',
                vmin=vmin,
                vmax=vmax)
            fig.colorbar(cax, ax=ax)
            ax.set_title(
                f'{activation.capitalize()} Activations for Layer {layer} in {file_name}')
            ax.set_xlabel('Neuronal activations')
            ax.set_ylabel('Timesteps')

    # Plotting the heatmap for the actual combined activations
    for layer in range(LAYERS):
        actual_combined = torch.abs(
            data[f'{activation_type}_activations'][:, layer, :]).cpu().numpy()
        ax = axes[5, layer]
        cax = ax.imshow(
            actual_combined,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            origin='lower',
            vmin=OUTPUT_ACTIVATION_LIMIT_LOWER,
            vmax=OUTPUT_ACTIVATION_LIMIT_UPPER)
        fig.colorbar(cax, ax=ax)
        ax.set_title(
            f'Actual Combined Activations (With Leaky ReLU) for Layer {layer} in {file_name}')
        ax.set_xlabel('Neuronal activations')
        ax.set_ylabel('Timesteps')

    # Compute and plot the summed activations for each layer
    for layer in range(LAYERS):
        summed_activations = (
            data[f'{activation_type}_forward_activations'] +
            data[f'{activation_type}_backward_activations'] +
            data[f'{activation_type}_lateral_activations']
        )[:, layer, :]
        summed_activations = summed_activations

        # With leaky_relu
        summed_activations_leaky = F.leaky_relu(
            summed_activations)
        summed_activations_leaky = torch.abs(
            summed_activations_leaky).cpu().numpy()

        # Without leaky_relu
        summed_activations = summed_activations.cpu().numpy()

        # Plotting the heatmap for the summed activations without leaky_relu
        ax = axes[3, layer]
        cax = ax.imshow(
            summed_activations,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            origin='lower',
            vmin=-50,
            vmax=30)
        fig.colorbar(cax, ax=ax)
        ax.set_title(
            f'Summed Activations (No Leaky ReLU) for Layer {layer} in {file_name}')
        ax.set_xlabel('Neuronal activations')
        ax.set_ylabel('Timesteps')

        # Plotting the heatmap for the summed activations with leaky_relu
        ax = axes[4, layer]
        cax = ax.imshow(
            summed_activations_leaky,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            origin='lower',
            vmin=OUTPUT_ACTIVATION_LIMIT_LOWER,
            vmax=OUTPUT_ACTIVATION_LIMIT_UPPER)
        fig.colorbar(cax, ax=ax)
        ax.set_title(
            f'Summed Activations (With Leaky ReLU) for Layer {layer} in {file_name}')
        ax.set_xlabel('Neuronal activations')
        ax.set_ylabel('Timesteps')

    # Finalizing the plots
    plt.tight_layout()
    plt.savefig(
        f"img/debug/activation_cancellations/heatmap_{activation_type}_{file_name.replace('.pt', '')}.png")


if __name__ == '__main__':

    plot_l2_norm_across_time('test_sample_3.pt', activation_type="correct")
    plot_l2_norm_across_time('test_sample_3.pt', activation_type="incorrect")

    plot_activation_heatmap('test_sample_3.pt', activation_type="correct")
    plot_activation_heatmap('test_sample_3.pt', activation_type="incorrect")

    file_names = [
        'test_sample_3.pt',
        'test_sample_2.pt',
        'test_sample_1.pt',
        'test_sample_4.pt',
        'test_sample_5.pt',
        'test_sample_6.pt',
        'test_sample_7.pt',
        'test_sample_8.pt',
        'test_sample_9.pt',
        'test_sample_11.pt']
    plot_cosine_similarity_multi_file(
        file_names, activation_type="correct")
    plot_cosine_similarity_multi_file(
        file_names, activation_type="incorrect")

    plot_cosine_similarity('test_sample_3.pt', activation_type="correct")
    plot_cosine_similarity('test_sample_3.pt', activation_type="incorrect")
