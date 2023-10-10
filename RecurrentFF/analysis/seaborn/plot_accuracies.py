import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust the order of file paths, labels, and colors
ADUSTED_FILE_PATHS = [
    # Batch Size 500
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_13_02.674-07_00.csv",
    # Batch Size 100
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_13_35.132-07_00.csv",
    # Sigmoid Nonlinearity
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_38_17.284-07_00.csv",
    # Stronger Backwards Weight Init
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_12_13.556-07_00.csv",
    # Identity Lateral Weight Init
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_11_49.849-07_00.csv"
]


sigma_slightly_less = 20

adjusted_labels = [
    "Batch Size 500",
    "Batch Size 100",
    "Sigmoid Activation Function",
    "Stronger Backwards Weight Init",
    "Identity Lateral Weight Init"
]
adjusted_colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]

# Initialize the plot
plt.figure(figsize=(14, 9))

# Seaborn style settings
sns.set_style("white")
sns.set_context("talk")

# Plot the datasets using the new order
for idx, file_path in enumerate(ADUSTED_FILE_PATHS):
    data = pd.read_csv(file_path)
    data = data[data["epoch"] <= 315]
    y_col = data["dataset: MNIST - train_acc"].values
    y_col_smoothed = gaussian_filter1d(y_col, sigma_slightly_less)
    y_col_upper_bound_smoothed = np.maximum(y_col, y_col_smoothed)

    sns.lineplot(x=data["epoch"], y=y_col_upper_bound_smoothed,
                 label=adjusted_labels[idx], color=adjusted_colors[idx], linewidth=3)
    sns.lineplot(x=data["epoch"], y=y_col, color=adjusted_colors[idx],
                 alpha=0.3, linestyle='--', linewidth=3)

# Remove spines (box around the figure)
sns.despine(left=False, bottom=False)

# Place the legend outside the plot area and remove its surrounding box
legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
# For the legend
for text in legend.get_texts():
    text.set_fontsize(32)
    text.set_fontweight(375)  # Set to a medium-thick font weight

# For axis labels and tick labels
plt.xlabel("Epoch", fontsize=36, fontweight=375)
plt.ylabel("Training Accuracy", fontsize=36, fontweight=375)
plt.xticks(fontsize=34, fontweight=375)
plt.yticks(fontsize=34, fontweight=375)

# Save to PDF
pdf_path = "./img/presentation/accuracy/accuracies.pdf"
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
