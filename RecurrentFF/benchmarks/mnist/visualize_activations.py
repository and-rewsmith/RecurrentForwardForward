import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_PATH = "./img/activation_heatmaps"

filenames = ["test_sample_1.pt", "test_sample_2.pt"]

for filename in filenames:
    identifier = filename.split(".")[0].split("_")[-1]
    tensors = torch.load(filename)

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

    for scenario in scenarios:
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
            print(f"\tStandard Deviation: {std_devs[layer].mean().item():.4f}")

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
            print(f"Layer {layer + 1}:")
            print(f"\tMean Activation: {means[layer].mean().item():.4f}")
            print(f"\tStandard Deviation: {std_devs[layer].mean().item():.4f}")

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
            f"{BASE_PATH}/means_{scenario}_{identifier}.png", dpi=300)

        # =========================================================================
        # Plot all timesteps absolute value activations
        # =========================================================================

        fig, axes = plt.subplots(timesteps, 1, figsize=(10, 5 * timesteps))

        for t in range(timesteps):
            timestep_tensor = loaded[t]
            timestep_tensor = torch.abs(timestep_tensor)

            df = pd.DataFrame(timestep_tensor.cpu())
            sns_heatmap = sns.heatmap(df, cmap='viridis', vmin=0, vmax=global_max, cbar_kws={
                'label': 'Activation Value'}, ax=axes[t])
            cbar = sns_heatmap.collections[0].colorbar
            cbar.set_label('Activation Value', fontsize=12)
            axes[t].set_title(f'Timestep {t}', fontsize=12)
            axes[t].set_xlabel('Neuron #', fontsize=12)
            axes[t].set_ylabel('Layer', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{BASE_PATH}/{scenario}_{identifier}.png", dpi=300)
