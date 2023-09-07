import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
import scipy


def clusters_to_communities(clusters):
    """
    Convert a list of cluster assignments to the community format.

    Parameters:
    - clusters: A list of cluster assignments for each node.

    Returns:
    - communities: A list of sets, where each set contains the nodes in a community.
    """
    communities = {}
    for node, cluster in enumerate(clusters):
        if cluster not in communities:
            communities[cluster] = set()
        communities[cluster].add(node)
    return list(communities.values())


def compute_elbow(eigenvectors, max_clusters=10):
    """
    Compute the sum of squared distances for different number of clusters.

    Parameters:
    - data: The input data for clustering.
    - max_clusters: The maximum number of clusters to check.

    Returns:
    - distortions: A list containing the sum of squared distances for each cluster number.
    """
    distortions = []
    for k in range(1, max_clusters+1):
        spectral_repr_k = eigenvectors[:, 1:k+1]
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(spectral_repr_k)
        # cdist computes distance between each pair of the two collections of inputs
        distortions.append(sum(np.min(
            scipy.spatial.distance.cdist(spectral_repr_k, kmeanModel.cluster_centers_, 'euclidean'), axis=1))
            / spectral_repr_k.shape[0])

    # Plotting the Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters+1), distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    # save fig
    plt.savefig("elbow.png")
    return distortions


def sanity_plot_graph():
    # plot adjacency matrix in matplotlib in graph form

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)

    # We can take a subset of the graph for clearer visualization:
    # Change 100 to a different number if needed
    H = G.subgraph(list(range(100)))

    pos = nx.spring_layout(H)
    plt.figure(figsize=(12, 12))
    nx.draw(H, pos, with_labels=False, node_size=10, width=0.2)

    # save fig
    plt.savefig("adjacency_matrix.png")


def plot_weight_magnitude_histogram(all_weights):
    # Plot a histogram of all weight magnitudes
    plt.figure(figsize=(10, 6))
    plt.hist(all_weights, bins=100, color='skyblue', edgecolor='black')
    plt.title('Histogram of Weight Magnitudes')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the histogram image (optional)
    plt.savefig("weight_histogram.png")


if __name__ == "__main__":
    weights = torch.load("weights_9-2.pth")

    for element in weights:
        print(element)

    adjacency_matrix = np.zeros((6000, 6000))

    layers = 3
    for i in range(layers):
        start_idx = i * 2000
        end_idx = (i + 1) * 2000

        backward_key = f"inner_layers.layers.{i}.forward_linear.weight"
        recurrent_key = f"inner_layers.layers.{i}.lateral_linear.weight"
        forward_key = f"inner_layers.layers.{i}.backward_linear.weight"

        # backward weights
        if i != layers - 1:
            adjacency_matrix[start_idx:end_idx, start_idx + 2000:end_idx +
                             2000] = np.abs(weights[forward_key].detach().cpu().numpy())

        # recurrent weights
        adjacency_matrix[start_idx:end_idx, start_idx:end_idx] = np.abs(
            weights[recurrent_key].detach().cpu().numpy())

        # forward weights
        if i != 0:
            adjacency_matrix[start_idx:end_idx, start_idx - 2000:end_idx -
                             2000] = np.abs(weights[backward_key].detach().cpu().numpy())

    # Flatten the adjacency matrix to get all weights as a 1D array
    all_weights = adjacency_matrix.flatten()

    # Find the threshold corresponding to the 25th percentile
    percent = 35
    threshold = np.percentile(all_weights, percent)

    # Print the threshold
    print(f"{percent} percentile threshold: {threshold}")

    # Set weights below the threshold to zero
    adjacency_matrix[adjacency_matrix < threshold] = 0

    # Calculate degree for each node (considering both in and out connections)
    degrees = np.sum(adjacency_matrix, axis=0) + \
        np.sum(adjacency_matrix, axis=1)

    # Create the diagonal degree matrix D
    D = np.diag(degrees)

    # Calculate the Laplacian L = D - A
    L = D - adjacency_matrix

    from numpy.linalg import eigh

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(L)

    # Print the smallest eigenvalue
    print(f"Smallest eigenvalue (lambda_0): {eigenvalues[0]}")

    # Print the second smallest eigenvalue
    print(f"Second smallest eigenvalue (lambda_1): {eigenvalues[1]}")

    # Fiedler vector
    fiedler_vector = eigenvectors[:, 1]

    # Print some components of the Fiedler vector (e.g., first 10 values)
    print(f"First 10 components of the Fiedler vector: {fiedler_vector[:10]}")

    # You can also analyze the Fiedler vector further:
    positive_nodes = np.where(fiedler_vector > 0)[0]
    negative_nodes = np.where(fiedler_vector < 0)[0]

    print(
        f"Number of nodes with positive values in Fiedler vector: {len(positive_nodes)}")
    print(
        f"Number of nodes with negative values in Fiedler vector: {len(negative_nodes)}")

    # Calculate optimal k using Elbow method
    distortions = compute_elbow(eigenvectors, max_clusters=15)
    # This finds the "elbow" in the distortion curve
    optimal_k = np.argmin(np.gradient(np.gradient(distortions))) + 1
    print(
        f"Optimal number of clusters (k) based on the Elbow method: {optimal_k}")

    # We'll use the optimal_k found using the elbow method for spectral clustering
    kmeans = KMeans(n_clusters=optimal_k)
    spectral_repr = eigenvectors[:, 1:optimal_k+1]
    clusters = kmeans.fit_predict(spectral_repr)

    # Let's print out the distribution of nodes in each cluster
    for cluster_num in range(optimal_k):
        print(
            f"Number of nodes in cluster {cluster_num}: {np.sum(clusters == cluster_num)}")

    print(clusters)

    G = nx.from_numpy_array(adjacency_matrix)

    communities = clusters_to_communities(clusters)
    modularity = nx.algorithms.community.quality.modularity(G, communities)

    print(f"Modularity: {modularity}")

    # Optionally: Visualize the clusters in a graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_color=clusters, cmap=plt.get_cmap(
        'jet'), node_size=10, width=0.2)
    plt.savefig("clusters.png")
