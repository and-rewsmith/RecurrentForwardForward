import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the parquet file into a dataframe
df = pd.read_parquet('converted_data.parquet')

# Compute mean activity by layer, is_label_showing, and activation_type
grouped = df.groupby(['layer_index', 'is_label_showing', 'activation_type'])[
    'activity'].mean().reset_index()

# Create a bar plot using seaborn
plt.figure(figsize=(15, 8))
sns.barplot(data=grouped, x='layer_index', y='activity',
            hue='activation_type', ci=None, palette='muted')

# Add a title and labels
plt.title('Mean Neural Activity by Layer Index')
plt.xlabel('Layer Index')
plt.ylabel('Mean Activity')
plt.legend(title='Activation Type', loc='upper right')

# Show the plot
plt.show()
