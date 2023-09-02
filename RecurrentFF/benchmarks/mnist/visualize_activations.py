import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_PATH = "./img/activation_heatmaps"

tensors = torch.load('test_sample_1.pt')

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
    plt.savefig(f"{BASE_PATH}/{scenario}.png", dpi=300)
