import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DATAFRAME_PATH = "./converted_data_926.parquet"
FILTER_CLASS = 3

# Load the parquet file
df = pd.read_parquet(DATAFRAME_PATH)

SAVE_BASEPATH = "./img/presentation/l2_norm_activations/"


def compute_l2_norm(grouped_df, column_name):
    """Compute L2 norm for a given column."""
    grouped_df[f'{column_name}_l2_norm'] = np.sqrt(grouped_df[column_name])
    return grouped_df.groupby(['image_timestep', 'layer_index', 'is_correct'])[f'{column_name}_l2_norm'].mean()\
        .reset_index()


def plot_activations_l2_over_time_from_df(df):
    target_array = np.array([FILTER_CLASS])
    df = df[df['label'].apply(lambda x: np.array_equal(x, target_array))]
    print("done with filtering")

    # Calculate squared activity for each component
    for component in ['activation',
                      'forward_activation_component', 'backward_activation_component', 'lateral_activation_component']:
        df[f'squared_{component}'] = df[component]**2

    # Group and calculate L2 norm for each group
    df_grouped = df.groupby(
        ['data_sample_id', 'image_timestep', 'layer_index', 'is_correct'])

    df_avg = pd.DataFrame()
    for component in ['squared_activation', 'squared_forward_activation_component',
                      'squared_backward_activation_component', 'squared_lateral_activation_component']:
        temp = compute_l2_norm(
            df_grouped[component].sum().reset_index(), component)
        if df_avg.empty:
            df_avg = temp
        else:
            df_avg = pd.merge(df_avg, temp, on=[
                              'image_timestep', 'layer_index', 'is_correct'])

    # Plot settings
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    # Color palette
    palette = sns.color_palette("colorblind")

    # Split data into positive and negative
    df_positive = df_avg[df_avg['is_correct']]
    df_negative = df_avg[~df_avg['is_correct']]

    for title_text, data in [("Negative Data", df_negative), ("Positive Data", df_positive)]:
        fig, axs = plt.subplots(5, 1, figsize=(10, 20))

        for idx, ax in enumerate(axs):
            layer_data = data[data['layer_index'] == idx]
            ax.plot(layer_data['image_timestep'],
                    layer_data['squared_activation_l2_norm'], label="Activation")
            ax.plot(layer_data['image_timestep'],
                    layer_data['squared_forward_activation_component_l2_norm'], label="Forward")
            ax.plot(layer_data['image_timestep'],
                    layer_data['squared_backward_activation_component_l2_norm'], label="Backward")
            ax.plot(layer_data['image_timestep'],
                    layer_data['squared_lateral_activation_component_l2_norm'], label="Lateral")
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('L2 Norm')
            ax.set_title(f'Layer {idx + 1}')

        fig.tight_layout()
        fig.suptitle(
            f'L2 Norm of Activation Components Over Time ({title_text})', y=1.02)
        fig.savefig(f'{SAVE_BASEPATH}/l2norm_activation_components_{title_text.replace(" ", "_").lower()}.pdf',
                    format='pdf', bbox_inches='tight')
        fig.savefig(f'{SAVE_BASEPATH}/l2norm_activation_components_{title_text.replace(" ", "_").lower()}.png',
                    format='png', bbox_inches='tight')


plot_activations_l2_over_time_from_df(df)
