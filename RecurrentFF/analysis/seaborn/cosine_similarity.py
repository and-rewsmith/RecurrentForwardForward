import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import seaborn as sns

BASIC_COMPARISONS = [
    ('forward_activation_component', 'backward_activation_component'),
    ('forward_activation_component', 'lateral_activation_component'),
    ('backward_activation_component', 'lateral_activation_component')
]

COMPLEX_COMPARISONS = [
    ('forward_activation_component',
        'backward_activation_component + lateral_activation_component'),
    ('backward_activation_component',
        'forward_activation_component + lateral_activation_component'),
    ('lateral_activation_component',
        'forward_activation_component + backward_activation_component')
]


def compute_cosine_similarity(df, is_correct):
    # Define the list of comparisons

    all_comparisons = BASIC_COMPARISONS + COMPLEX_COMPARISONS

    df = df[df['is_correct'] == is_correct]
    df = df[['image_timestep', 'neuron index', 'forward_activation_component',
            'backward_activation_component', 'lateral_activation_component']]

    forward_tensor = df.pivot(index='image_timestep', columns='neuron index',
                              values='forward_activation_component').values
    backward_tensor = df.pivot(index='image_timestep', columns='neuron index',
                               values='backward_activation_component').values
    lateral_tensor = df.pivot(index='image_timestep', columns='neuron index',
                              values='lateral_activation_component').values

    # concatenate the three tensors
    concatenated_tensor = np.concatenate(
        (forward_tensor, backward_tensor, lateral_tensor), axis=0)

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

        activations2 = None
        if '+' in act2:
            act2_parts = act2.split('+')
            act2_parts = [a.strip() for a in act2_parts]

            activations2_1 = component_mappings[act2_parts[0]]
            activations2_2 = component_mappings[act2_parts[1]]
            activations2 = activations2_1 + activations2_2
        else:
            activations2 = component_mappings[act2]

        cos_sim = [
            cosine_similarity(
                activations1[i].reshape(1, -1),
                activations2[i].reshape(1, -1))[0][0] for i in range(
                activations1.shape[0])]

        cos_sim_results[(act1, act2)] = cos_sim

    return cos_sim_results


def plot_cosine_similarity(df):

    sns.set_style("whitegrid")  # Set Seaborn style

    df = df.drop(columns=['image', 'dataset'])

    df_grouped = df.groupby(
        ['layer_index', 'neuron index', 'image_timestep', 'is_correct']).mean().reset_index()

    n_layers = df['layer_index'].nunique()

    # We'll generate a list to store all the data to be plotted
    plot_data = []

    for layer in df['layer_index'].unique():
        df_layer = df_grouped[df_grouped['layer_index'] == layer]
        cos_sims_pos = compute_cosine_similarity(df_layer, True)
        cos_sims_neg = compute_cosine_similarity(df_layer, False)

        for comparison, cos_sim in cos_sims_pos.items():
            comp1 = comparison[0].split("_")[0]
            comp2 = comparison[1].split("_")[0] if "+" not in comparison[1] else "(" + comparison[1].split(
                "+")[0].split("_")[0] + " + " + comparison[1].split("+")[1].split("_")[0] + ")"
            for i, val in enumerate(cos_sim):
                plot_data.append({
                    'layer': layer,
                    'timestep': i,
                    'similarity': val,
                    'comparison': f'{comp1} vs {comp2}',
                    'type': 'Basic' if comparison in BASIC_COMPARISONS else 'Complex',
                    'is_pos_data': True
                })

        for comparison, cos_sim in cos_sims_neg.items():
            comp1 = comparison[0].split("_")[0]
            comp2 = comparison[1].split("_")[0] if "+" not in comparison[1] else "(" + comparison[1].split(
                "+")[0].split("_")[0] + " + " + comparison[1].split("+")[1].split("_")[0] + ")"
            for i, val in enumerate(cos_sim):
                plot_data.append({
                    'layer': layer,
                    'timestep': i,
                    'similarity': val,
                    'comparison': f'{comp1} vs {comp2}',
                    'type': 'Basic' if comparison in BASIC_COMPARISONS else 'Complex',
                    'is_pos_data': False
                })

    # Convert the list to a DataFrame
    plot_df = pd.DataFrame(plot_data)
    # Replace is_pos_data values for clarity in titles
    plot_df['is_pos_data'] = plot_df['is_pos_data'].replace(
        {True: 'Positive', False: 'Negative'})

    # Separate the data into basic and complex
    df_basic = plot_df[plot_df['type'] == 'Basic']
    df_complex = plot_df[plot_df['type'] == 'Complex']

    # Plotting for basic comparisons
    g_basic = sns.FacetGrid(df_basic, col="layer",
                            row="is_pos_data", height=5, aspect=1.2)
    g_basic = g_basic.map(sns.lineplot, 'timestep',
                          'similarity', 'comparison', palette='tab10')
    g_basic.add_legend()

    # Set titles for basic plot
    g_basic.set_titles(
        col_template='Layer {col_name}', row_template='{row_name} Data')
    g_basic.set_axis_labels('Time Step', 'Cosine Similarity')

    g_basic.savefig("img/presentation/2E_cosine_sim/basic_comparisons.png")
    g_basic.savefig("img/presentation/2E_cosine_sim/basic_comparisons.pdf",
                    format='pdf', bbox_inches='tight')

    # Plotting for complex comparisons
    g_complex = sns.FacetGrid(df_complex, col="layer",
                              row="is_pos_data", height=5, aspect=1.2)
    g_complex = g_complex.map(sns.lineplot, 'timestep',
                              'similarity', 'comparison', palette='tab10')
    g_complex.add_legend()

    # Set titles for complex plot
    g_complex.set_titles(
        col_template='Layer {col_name}', row_template='{row_name} Data')
    g_complex.set_axis_labels('Time Step', 'Cosine Similarity')

    g_complex.savefig("img/presentation/2E_cosine_sim/complex_comparisons.png")
    g_complex.savefig("img/presentation/2E_cosine_sim/complex_comparisons.pdf",
                      format='pdf', bbox_inches='tight')


df = pd.read_parquet('~/Downloads/dataframe_9-19.parquet')
plot_cosine_similarity(df)
