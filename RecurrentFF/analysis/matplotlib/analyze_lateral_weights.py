import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

# Load the weights
weights = torch.load("weights_9-2.pth")

matrix_size = 6000
layers = 3

# 1. Extract the lateral connectivity matrices from the weights.
lateral_matrices = []
for i in range(layers):
    recurrent_key = f"inner_layers.layers.{i}.lateral_linear.weight"
    lateral_matrix = np.abs(weights[recurrent_key].detach().cpu().numpy())
    lateral_matrices.append(lateral_matrix)

# 2. Compute the eigenvalues of these matrices.
eigenvalues_list = [eig(matrix)[0] for matrix in lateral_matrices]

# 3. Analyze the real parts of the eigenvalues to determine the stability of the system.
for idx, eigenvalues in enumerate(eigenvalues_list):
    unstable_modes = np.sum(np.real(eigenvalues) > 1)
    print(f"For lateral matrix {idx+1}:")
    print(
        f"Number of modes with real parts of their eigenvalues above 1 (unstable modes): {unstable_modes}")
    if unstable_modes == 0:
        print("The network is stable for this layer.\n")
    else:
        print("The network is unstable for this layer.\n")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=10, c='blue')
    plt.axvline(x=1, color='red', linestyle='--')  # Stability threshold
    plt.title(f'Eigenvalues for Lateral Matrix {idx+1}')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    plt.savefig(f"img/weight_distributions/eigenvalues_layer_{idx+1}.png")
