import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DATAFRAME_PATH = "./converted_data_spike_elim_rescaled_2_250.parquet"
FILTER_CLASS = 3

# Load the parquet file
df = pd.read_parquet(DATAFRAME_PATH)

SAVE_BASEPATH = "./img/presentation/l2_norm_activations/"


def plot_activations_l2_over_time_from_df(df):
    # Compute the squared activity for each neuron
    df['squared_activity'] = df['activation']**2

    target_array = np.array([FILTER_CLASS])
    df = df[df['label'].apply(lambda x: np.array_equal(x, target_array))]
    print("done with filtering")

    # Group and calculate L2 norm for each group
    df_grouped = df.groupby(
        ['data_sample_id', 'image_timestep', 'layer_index', 'is_correct']
    )['squared_activity'].sum().reset_index()

    df_grouped['l2_norm'] = np.sqrt(df_grouped['squared_activity'])

    # Average away the 'data_sample_id' dimension
    df_avg = df_grouped.groupby(
        ['image_timestep', 'layer_index', 'is_correct']
    )['l2_norm'].mean().reset_index()

    # Plot settings
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")  # Changed to remove the gridlines

    # Color palette
    palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

    # Split data into positive and negative
    df_positive = df_avg[df_avg['is_correct']]
    df_negative = df_avg[~df_avg['is_correct']]

    for label, data in [("Negative Data", df_negative), ("Positive Data", df_positive)]:
        g = sns.relplot(
            data=data, x='image_timestep', y='l2_norm', hue='layer_index',
            kind='line', height=4, aspect=2, palette=palette, legend="full",
            linewidth=2
        )
        g._legend.set_bbox_to_anchor((1.05, 0.5))

        g.set(ylim=(0, 80))
        g.set(xlim=(0, 25))

        # Enhancements
        g.set_axis_labels("Time", "L2 Norm")
        g.set_titles(label)

        g.despine()  # Remove top and right spines

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(
            f'{SAVE_BASEPATH}/l2norm_over_time_truncated_{label.replace(" ", "_").lower()}.pdf',
            format='pdf', bbox_inches='tight'
        )
        plt.savefig(
            f'{SAVE_BASEPATH}/l2norm_over_time_truncated_filtered_{label.replace(" ", "_").lower()}.png',
            format='png', bbox_inches='tight'
        )


plot_activations_l2_over_time_from_df(df)
