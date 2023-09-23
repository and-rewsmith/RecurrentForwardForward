import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def compute_cosine_similarity(df):
    # Define the list of comparisons
    basic_comparisons = [
        ('forward_activation_component', 'backward_activation_component'),
        ('forward_activation_component', 'lateral_activation_component'),
        ('backward_activation_component', 'lateral_activation_component')
    ]

    complex_comparisons = [
        ('forward_activation_component',
         'backward_activation_component + lateral_activation_component'),
        ('backward_activation_component',
         'forward_activation_component + lateral_activation_component'),
        ('lateral_activation_component',
         'forward_activation_component + backward_activation_component')
    ]

    all_comparisons = basic_comparisons + complex_comparisons

    df = df[['image_timestep', 'neuron index', 'forward_activation_component',
            'backward_activation_component', 'lateral_activation_component']]

    # print(df.shape)
    # print(df.columns)
    # print(df[['forward_activation_component', 'backward_activation_component',
    #       'lateral_activation_component']].values.shape)
    # print(df[['forward_activation_component', 'backward_activation_component',
    #           'lateral_activation_component']].columns)
    # input()

    forward_tensor = df.pivot(index='image_timestep', columns='neuron index',
                              values='forward_activation_component').values
    backward_tensor = df.pivot(index='image_timestep', columns='neuron index',
                               values='backward_activation_component').values
    lateral_tensor = df.pivot(index='image_timestep', columns='neuron index',
                              values='lateral_activation_component').values

    # concatenate the three tensors
    concatenated_tensor = np.concatenate(
        (forward_tensor, backward_tensor, lateral_tensor), axis=0)

    # print(forward_tensor.shape)
    # print(backward_tensor.shape)
    # print(lateral_tensor.shape)
    # print(concatenated_tensor.shape)
    # input()

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=5)
    pca = pca.fit(concatenated_tensor)

    forward_tensor = pca.transform(forward_tensor)
    backward_tensor = pca.transform(backward_tensor)
    lateral_tensor = pca.transform(lateral_tensor)

    component_mappings = {"forward_activation_component": forward_tensor,
                          "backward_activation_component": backward_tensor,
                          "lateral_activation_component": lateral_tensor}

    # Calculate cosine similarity for each comparison
    cos_sim_results = {}
    for act1, act2 in all_comparisons:
        activations1 = component_mappings[act1]
        print(activations1.shape)

        activations2 = None
        if '+' in act2:
            act2_parts = act2.split('+')
            act2_parts = [a.strip() for a in act2_parts]

            activations2_1 = component_mappings[act2_parts[0]]
            activations2_2 = component_mappings[act2_parts[1]]
            activations2 = activations2_1 + activations2_2
            print(activations2.shape)
        else:
            activations2 = component_mappings[act2]
            print(activations2.shape)

        cos_sim = [
            cosine_similarity(
                activations1[i].reshape(1, -1),
                activations2[i].reshape(1, -1))[0][0] for i in range(
                activations1.shape[0])]

        # print(cos_sim)
        # input()

        cos_sim_results[(act1, act2)] = cos_sim

    return cos_sim_results


def plot_cosine_similarity(df, is_correct):

    # Filter the dataframe based on is_correct
    df_filtered = df[df['is_correct'] == is_correct]
    if 'image' in df.columns:
        df_filtered = df_filtered.drop(columns=['image', 'dataset'])

    # print(df_filtered.shape)

    # Group by required dimensions and compute mean
    df_grouped = df_filtered.groupby(
        ['layer_index', 'neuron index', 'image_timestep']).mean().reset_index()

    # Number of unique layers
    n_layers = df['layer_index'].nunique()

    # Plot settings
    fig, axes = plt.subplots(n_layers, 2, figsize=(15, 5 * n_layers))

    for layer, (ax1, ax2) in zip(df['layer_index'].unique(), axes):
        df_layer = df_grouped[df_grouped['layer_index'] == layer]
        cos_sims = compute_cosine_similarity(df_layer)

        timestep = 0
        for comparison, cos_sim in cos_sims.items():
            ax = None
            if timestep < 3:
                ax = ax1
            else:
                ax = ax2

            # print("image timestep")
            # print(df_layer['image_timestep'].shape)
            # input()

            ax.plot([i for i in range(0, len(cos_sim))], cos_sim,
                    label=f'{comparison[0].split("_")[0]} vs {comparison[1].split("_")[0]}')
            timestep += 1

        ax1.set_title(f'Layer {layer} - Basic Comparisons')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Cosine Similarity')
        ax1.legend()

        ax2.set_title(f'Layer {layer} - Complex Comparisons')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Cosine Similarity')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f"img/presentation/cosine_sim/{'correct' if is_correct else 'incorrect'}.png")


df = pd.read_parquet('~/Downloads/dataframe_9-19.parquet')
plot_cosine_similarity(df, True)
plot_cosine_similarity(df, False)
