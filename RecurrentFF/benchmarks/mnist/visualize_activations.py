import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tensors = torch.load('test_sample_1.pt')

scenario = "correct_activations"
loaded = tensors[scenario]
print(loaded.shape)
timesteps = loaded.shape[0]

fig, axes = plt.subplots(timesteps, 1, figsize=(10, 5 * timesteps))

for t in range(timesteps):
    df = pd.DataFrame(loaded[t].cpu().numpy())
    sns_heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={
                              'label': 'Activation Value'}, ax=axes[t])
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label('Activation Value', fontsize=12)
    axes[t].set_title(f'Timestep {t}', fontsize=12)
    axes[t].set_xlabel('Neuron #', fontsize=12)
    axes[t].set_ylabel('Layer', fontsize=12)

plt.tight_layout()
plt.savefig(f"{scenario}.png", dpi=300)
