import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

sigma_slightly_less = 20

# Adjust the order of file paths, labels, and colors
adjusted_file_paths = [
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_13_02.674-07_00.csv",  # Batch Size 500
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_13_35.132-07_00.csv",  # Batch Size 100
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_38_17.284-07_00.csv",  # Sigmoid Nonlinearity
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_12_13.556-07_00.csv",  # Stronger Backwards Weight Init
    "/home/andrew/Downloads/figures/wandb_export_2023-09-26T20_11_49.849-07_00.csv"   # Identity Lateral Weight Init
]
adjusted_labels = [
    "Batch Size 500",
    "Batch Size 100",
    "Sigmoid Nonlinearity",
    "Stronger Backwards Weight Init",
    "Identity Lateral Weight Init"
]
adjusted_colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]

# Initialize the plot without gridlines
plt.figure(figsize=(14, 9))

# Plot the datasets with the adjusted smoothing, using the new order
for idx, file_path in enumerate(adjusted_file_paths):
    data = pd.read_csv(file_path)
    data = data[data["epoch"] <= 315]
    y_col = data["dataset: MNIST - train_acc"].values
    y_col_smoothed = gaussian_filter1d(y_col, sigma_slightly_less)
    y_col_upper_bound_smoothed = np.maximum(y_col, y_col_smoothed)
    plt.plot(data["epoch"], y_col_upper_bound_smoothed, label=adjusted_labels[idx], color=adjusted_colors[idx], linewidth=2)
    plt.plot(data["epoch"], y_col, color=adjusted_colors[idx], alpha=0.3, linestyle='--')

# Finalize the plot with the updated settings
plt.legend(loc="lower right", fontsize=18)  # Increased font size for the legend
plt.title("Training Accuracy across Configurations", fontsize=24)  # Increased font size for the title
plt.xlabel("Epoch", fontsize=20)  # Increased font size for the x-axis label
plt.ylabel("Training Accuracy", fontsize=20)  # Increased font size for the y-axis label
plt.xticks(fontsize=18)  # Increased font size for x-ticks
plt.yticks(fontsize=18)  # Increased font size for y-ticks
plt.grid(False)  # Ensure gridlines are turned off
plt.tight_layout()

# Save to PDF again
pdf_path = "./img/presentation/2F_accuracy/accuracies.pdf"
plt.savefig(pdf_path, format='pdf')
plt.close()