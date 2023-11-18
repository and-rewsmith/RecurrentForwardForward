import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

BASE_INPUT_PATH = "./artifacts/activations"
BASE_OUT_PATH = "./img/debug/activation_heatmaps"
SCENARIOS = ["incorrect_activations", "correct_activations"]
FILENAMES = ["test_sample_4.pt", "test_sample_2.pt", "test_sample_3.pt"]
MAX_ACTIVATION_DIVISOR = 4


def plot_mean_stddev():
    running_sum_activations = None
    for filename in FILENAMES:
        tensors = torch.load(f"{BASE_INPUT_PATH}/{filename}")

        for scenario in SCENARIOS:
            loaded = torch.abs(tensors[scenario])

            if running_sum_activations is None:
                running_sum_activations = loaded
            else:
                running_sum_activations += loaded

    print("=====All Data=====")

    running_sum_activations = running_sum_activations

    # Compute mean and std along the timestep dimension for each layer
    means = running_sum_activations.mean(dim=0)
    std_devs = running_sum_activations.std(dim=0)

    # Print out the means and standard deviations for each layer
    for layer in range(means.shape[0]):
        print(f"Layer {layer + 1}:")
        print(f"\tMean Activation: {means[layer].mean().item():.4f}")
        print(f"\tStandard Deviation: {std_devs[layer].mean().item():.4f}")

    fig, axes = plt.subplots(4, 1, figsize=(10, 5 * 4))

    # Plot Std Dev
    df = pd.DataFrame(std_devs.cpu().numpy())
    sns_heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={
        'label': 'Average Std Dev'}, ax=axes[0])
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label('Average Std Dev', fontsize=12)
    axes[0].set_title(
        'Average Neuron Std Dev (across [batch, timesteps, layers])',
        fontsize=12)
    axes[0].set_xlabel('Neuron #', fontsize=12)
    axes[0].set_ylabel('Layer', fontsize=12)

    # Plot Std Dev Lower Range
    df = pd.DataFrame(std_devs.cpu().numpy())
    sns_heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={
        'label': 'Average Std Dev'}, ax=axes[1], vmin=0, vmax=2)
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label('Average Std Dev', fontsize=12)
    axes[1].set_title(
        'Average Neuron Std Dev (across [batch, timesteps, layers])',
        fontsize=12)
    axes[1].set_xlabel('Neuron #', fontsize=12)
    axes[1].set_ylabel('Layer', fontsize=12)

    # Plot Mean
    df = pd.DataFrame(means.cpu().numpy())
    sns_heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={
        'label': 'Average Activation Value'}, ax=axes[2])
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label(
        'Average Activation Value', fontsize=12)
    axes[2].set_title(
        'Average Abs Neuron Activation (across [batch, timesteps, layers])',
        fontsize=12)
    axes[2].set_xlabel('Neuron #', fontsize=12)
    axes[2].set_ylabel('Layer', fontsize=12)

    # Plot Mean Lower Range
    sns_heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={
        'label': 'Average Activation Value'}, ax=axes[3], vmin=0, vmax=2)
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label(
        'Average Activation Value', fontsize=12)
    axes[3].set_title(
        'Average Abs Neuron Activation (across [batch, timesteps, layers])',
        fontsize=12)
    axes[3].set_xlabel('Neuron #', fontsize=12)
    axes[3].set_ylabel('Layer', fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{BASE_OUT_PATH}/all_means_std_dev.png", dpi=300)


def plot_activations_over_timesteps():
    for filename in FILENAMES:
        identifier = filename.split(".")[0].split("_")[-1]
        tensors = torch.load(f"{BASE_INPUT_PATH}/{filename}")

        print(f"=====Filename: {filename}=====")

        scenarios = ["incorrect_activations", "correct_activations"]

        global_max = 0
        for scenario in scenarios:
            loaded = tensors[scenario]
            timesteps = loaded.shape[0]

            for t in range(timesteps):
                timestep_tensor = loaded[t]
                timestep_tensor = torch.abs(timestep_tensor)

                max_from_timestep = torch.max(timestep_tensor).item()
                if max_from_timestep > global_max:
                    global_max = max_from_timestep

        global_max = global_max // MAX_ACTIVATION_DIVISOR

        for scenario in scenarios:
            print(f"---------Scenario: {scenario}---------")

            loaded = tensors[scenario]
            timesteps = loaded.shape[0]

            # =========================================================================
            # Compute stats and plot for activations
            # =========================================================================

            # Compute mean and std along the timestep dimension for each layer
            means = loaded.mean(dim=0)
            std_devs = loaded.std(dim=0)

            # Print out the means and standard deviations for each layer
            for layer in range(means.shape[0]):
                print(f"Layer {layer + 1}:")
                print(f"\tMean Activation: {means[layer].mean().item():.4f}")
                print(
                    f"\tStandard Deviation: {std_devs[layer].mean().item():.4f}")
                print()

            # Convert to DataFrame for easy plotting
            df = pd.DataFrame(means.cpu().numpy())

            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(10, 5 * 2))
            sns_heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={
                'label': 'Average Activation Value'}, ax=axes[0])
            cbar = sns_heatmap.collections[0].colorbar
            cbar.set_label('Average Activation Value', fontsize=12)
            axes[0].set_title(
                'Average Neuron Activation', fontsize=12)
            axes[0].set_xlabel('Neuron #', fontsize=12)
            axes[0].set_ylabel('Layer', fontsize=12)

            # =========================================================================
            # Compute stats and plot for abs activations
            # =========================================================================

            loaded_abs = torch.abs(loaded)

            # Compute mean and std along the timestep dimension for each layer
            means = loaded_abs.mean(dim=0)
            std_devs = loaded_abs.std(dim=0)

            # Print out the means and standard deviations for each layer
            for layer in range(means.shape[0]):
                print(f"(Abs) Layer {layer + 1}:")
                print(f"\tMean Activation: {means[layer].mean().item():.4f}")
                print(
                    f"\tStandard Deviation: {std_devs[layer].mean().item():.4f}")
                print()

            # Convert to DataFrame for easy plotting
            df = pd.DataFrame(means.cpu().numpy())

            # Plot
            sns_heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={
                'label': 'Average Activation Value'}, ax=axes[1])
            cbar = sns_heatmap.collections[0].colorbar
            cbar.set_label(
                'Average Activation Value (absolute value)', fontsize=12)
            axes[1].set_title(
                'Average Abs Neuron Activation', fontsize=12)
            axes[1].set_xlabel('Neuron #', fontsize=12)
            axes[1].set_ylabel('Layer', fontsize=12)
            plt.tight_layout()
            plt.savefig(
                f"{BASE_OUT_PATH}/means_{scenario}_{identifier}.png", dpi=300)

            # =========================================================================
            # Plot all timesteps absolute value activations
            # =========================================================================

            fig, axes = plt.subplots(timesteps, 1, figsize=(10, 5 * timesteps))

            for t in range(timesteps):
                timestep_tensor = loaded[t]
                timestep_tensor = torch.abs(timestep_tensor)

                df = pd.DataFrame(timestep_tensor.cpu())
                sns_heatmap = sns.heatmap(
                    df, cmap='viridis', vmin=0, vmax=global_max, cbar_kws={
                        'label': 'Activation Value'}, ax=axes[t])
                cbar = sns_heatmap.collections[0].colorbar
                cbar.set_label('Activation Value', fontsize=12)
                axes[t].set_title(f'Timestep {t}', fontsize=12)
                axes[t].set_xlabel('Neuron #', fontsize=12)
                axes[t].set_ylabel('Layer', fontsize=12)

            plt.tight_layout()
            plt.savefig(
                f"{BASE_OUT_PATH}/{scenario}_{identifier}.png",
                dpi=300)


def plot_activations_over_time():
    for filename in FILENAMES:
        identifier = filename.split(".")[0].split("_")[-1]
        tensors = torch.load(f"{BASE_INPUT_PATH}/{filename}")

        for scenario in SCENARIOS:
            loaded = tensors[scenario]

            # Compute L2 norms over neurons
            l2_norms = torch.norm(loaded, p=2, dim=-1)

            plt.figure(figsize=(12, 8))
            for layer in range(l2_norms.shape[-1]):
                plt.plot(l2_norms[:, layer].cpu().numpy(),
                         label=f"Layer {layer + 1}")

            # Plotting the average L2 norm over all layers
            avg_over_all_layers = l2_norms.mean(dim=-1)
            plt.plot(avg_over_all_layers.cpu().numpy(),
                     label="Average Over All Layers", linestyle='--')

            plt.title(
                f'L2 Norm of Activations Over Time ({scenario} - {identifier})')
            plt.xlabel('Time')
            plt.ylabel('L2 Norm')
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"{BASE_OUT_PATH}/l2norm_over_time_{scenario}_{identifier}.png",
                dpi=300)
            plt.close()


def plot_sparsity_over_time(threshold=0.01):
    for filename in FILENAMES:
        identifier = filename.split(".")[0].split("_")[-1]
        tensors = torch.load(f"{BASE_INPUT_PATH}/{filename}")

        for scenario in SCENARIOS:
            loaded = tensors[scenario]

            # Calculate fraction of values with abs value below the threshold
            is_sparse = (loaded.abs() < threshold).float()
            # Fraction of sparse values over neurons
            sparsity_fraction = is_sparse.mean(dim=-1)

            plt.figure(figsize=(12, 8))
            for layer in range(sparsity_fraction.shape[-1]):
                plt.plot(sparsity_fraction[:, layer].cpu().numpy(),
                         label=f"Layer {layer + 1}")

            # Plotting the average sparsity fraction over all layers
            avg_over_all_layers = sparsity_fraction.mean(dim=-1)
            plt.plot(avg_over_all_layers.cpu().numpy(),
                     label="Average Over All Layers", linestyle='--')
            plt.ylim(0, 0.2)

            plt.title(f'Sparsity Over Time ({scenario} - {identifier})')
            plt.xlabel('Time')
            plt.ylabel('Sparsity Fraction')
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"{BASE_OUT_PATH}/sparsity_over_time_{scenario}_{identifier}.png",
                dpi=300)
            plt.close()


def plot_activation_percentiles_over_time(percentiles=[10, 25, 50, 75, 90]):
    for filename in FILENAMES:
        identifier = filename.split(".")[0].split("_")[-1]
        tensors = torch.load(f"{BASE_INPUT_PATH}/{filename}")

        for scenario in SCENARIOS:
            loaded = tensors[scenario].abs()

            num_layers = loaded.shape[1]
            percentile_values = {}

            # Calculate percentiles for each layer and time step
            for layer in range(num_layers):
                for percentile in percentiles:
                    # Note the change in axis for percentile computation
                    values = np.percentile(
                        loaded[:, layer, :].cpu().numpy(), percentile, axis=-1)
                    percentile_values[(layer, percentile)] = values

            fig, axes = plt.subplots(
                num_layers, 1, figsize=(12, 6 * num_layers))

            for layer, ax in enumerate(axes):
                for percentile in percentiles:
                    values = percentile_values[(layer, percentile)]
                    # Displaying the latest time value for brevity
                    ax.plot(
                        values, label=f"{percentile}th Percentile")

                ax.set_title(f'Layer {layer + 1}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Activation Magnitude')
                # limit y to 0, 1.5
                ax.set_ylim(0, 1.5)
                ax.legend()

            plt.suptitle(
                f'Activation Magnitudes by Percentile Over Time ({scenario} - {identifier})')
            plt.tight_layout()
            plt.savefig(
                f"{BASE_OUT_PATH}/activation_percentiles_over_time_{scenario}_{identifier}.png",
                dpi=300)
            plt.close()


def plot_activations_over_timesteps_limit():
    for filename in FILENAMES:
        identifier = filename.split(".")[0].split("_")[-1]
        tensors = torch.load(f"{BASE_INPUT_PATH}/{filename}")

        print(f"=====Filename: {filename}=====")

        scenarios = ["incorrect_activations", "correct_activations"]

        global_max = 0
        for scenario in scenarios:
            loaded = tensors[scenario]
            timesteps = loaded.shape[0]

            for t in range(timesteps):
                timestep_tensor = loaded[t]
                timestep_tensor = torch.abs(timestep_tensor)

                max_from_timestep = torch.max(timestep_tensor).item()
                if max_from_timestep > global_max:
                    global_max = max_from_timestep

        global_max = global_max // MAX_ACTIVATION_DIVISOR

        for scenario in scenarios:
            print(f"---------Scenario: {scenario}---------")

            loaded = tensors[scenario]
            timesteps = loaded.shape[0]

            # =========================================================================
            # Plot timesteps 15 through 21 absolute value activations
            # =========================================================================

            fig, axes = plt.subplots(1, 7, figsize=(18, 5))

            for t in range(15, 22):
                timestep_tensor = loaded[t]
                timestep_tensor = torch.abs(timestep_tensor)

                df = pd.DataFrame(timestep_tensor.cpu())
                sns_heatmap = sns.heatmap(
                    df, cmap='viridis', vmin=0, vmax=global_max, cbar=t == 21,
                    cbar_kws={'label': 'Activation Value'}, ax=axes[t-15])

                if t == 21:  # Only label the last image
                    cbar = sns_heatmap.collections[0].colorbar
                    cbar.set_label('Activation Value', fontsize=14)
                axes[t-15].set_title(f'Timestep {t}', fontsize=14)
                axes[t-15].set_xlabel('Neuron #', fontsize=14)
                axes[t-15].set_ylabel('')

                axes[t-15].invert_yaxis()

            scenario_legible = None
            if scenario == "incorrect_activations":
                scenario_legible = "Positive Data"
            else:
                scenario_legible = "Negative Data"

            # set title of plot
            plt.suptitle(
                f'Activations Over Time ({scenario_legible})',
                fontsize=16)
            plt.tight_layout()

            # save as very high quality png
            plt.savefig(
                f"./img/presentation/heatmaps/{scenario}_{identifier}.png",
                dpi=1200)
            plt.savefig(
                f"./img/presentation/heatmaps/{scenario}_{identifier}.pdf",
                bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    plot_mean_stddev()
    plot_activations_over_timesteps()
    plot_activations_over_time()
    plot_sparsity_over_time()
    plot_activation_percentiles_over_time()
    plot_activations_over_timesteps_limit()
