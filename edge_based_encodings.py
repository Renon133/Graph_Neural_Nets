import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def compute_kernel_based_edge_encodings(graph, alpha=0.5):
    """
    Computes edge structural encodings using a kernel-based approach based on shortest path distances.

    Parameters:
        graph (networkx.Graph): The input graph.
        alpha (float): Hyperparameter controlling the decay rate of the kernel function.

    Returns:
        dict: A dictionary where keys are edges (tuples) and values are kernel-based encodings.
    """
    # Compute shortest path distances
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    edge_encodings = {}
    for u, v in graph.edges:
        # Kernel function k(u, v) = exp(-alpha * d(u, v))
        d_uv = shortest_paths[u][v]  # Shortest path distance between u and v
        edge_encodings[(u, v)] = np.exp(-alpha * d_uv)

    return edge_encodings

def compute_residual_edge_channel(graph):
    """
    Computes residual edge channels based on adjacency and shortest path matrices.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        np.ndarray: Residual edge channel matrix.
    """
    # Adjacency matrix
    adjacency_matrix = nx.to_numpy_array(graph)

    # Shortest path matrix
    shortest_path_matrix = np.zeros_like(adjacency_matrix)
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    for i, u in enumerate(graph.nodes):
        for j, v in enumerate(graph.nodes):
            if v in shortest_paths[u]:
                shortest_path_matrix[i, j] = shortest_paths[u][v]

    # Residual edge channel as the sum of adjacency and shortest path matrices
    residual_edge_channel = adjacency_matrix + shortest_path_matrix

    return residual_edge_channel

def compute_attention_based_encodings(graph, kernel_encodings, d_k=1.0):
    """
    Computes attention-based edge encodings using kernel-based values.

    Parameters:
        graph (networkx.Graph): The input graph.
        kernel_encodings (dict): Kernel-based edge encodings.
        d_k (float): Scaling factor for attention.

    Returns:
        dict: A dictionary where keys are edges and values are attention scores.
    """
    # Collect all kernel values and scale them
    scaled_kernel_values = np.array([k_uv / np.sqrt(d_k) for k_uv in kernel_encodings.values()])

    # Apply softmax across all edges
    normalized_attention = softmax(scaled_kernel_values)

    # Map normalized attention scores back to edges
    attention_scores = {
        edge: normalized_attention[i]
        for i, edge in enumerate(kernel_encodings.keys())
    }

    return attention_scores


def visualize_edge_encodings(graph, edge_encodings, title):
    """
    Visualizes the graph with edge encodings as labels.

    Parameters:
        graph (networkx.Graph): The input graph.
        edge_encodings (dict): A dictionary of edge encodings.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')

    for edge, encoding in edge_encodings.items():
        x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        plt.text(x, y, s=f"{encoding:.2f}", bbox=dict(facecolor='white', alpha=0.5), fontsize=8)

    plt.title(title)
    plt.show()

def compare_edge_encodings(graph, kernel_encodings, residual_channel, attention_encodings):
    """
    Compares kernel-based, residual, and attention-based edge encodings visually.

    Parameters:
        graph (networkx.Graph): The input graph.
        kernel_encodings (dict): Kernel-based edge encodings.
        residual_channel (np.ndarray): Residual edge channel matrix.
        attention_encodings (dict): Attention-based edge encodings.
    """
    visualize_edge_encodings(graph, kernel_encodings, "Kernel-Based Edge Encodings")

    # Convert residual channel matrix to edge encodings for visualization
    residual_edge_encodings = {
        (u, v): residual_channel[i, j]
        for i, u in enumerate(graph.nodes)
        for j, v in enumerate(graph.nodes)
        if graph.has_edge(u, v)
    }
    visualize_edge_encodings(graph, residual_edge_encodings, "Residual Edge Channel Encodings")

    visualize_edge_encodings(graph, attention_encodings, "Attention-Based Edge Encodings")

# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (2, 4)])

    # Compute kernel-based edge encodings
    kernel_encodings = compute_kernel_based_edge_encodings(G, alpha=0.5)

    # Compute residual edge channel
    residual_channel = compute_residual_edge_channel(G)

    # Compute attention-based edge encodings
    attention_encodings = compute_attention_based_encodings(G, kernel_encodings)

    # Compare edge encoding methods
    compare_edge_encodings(G, kernel_encodings, residual_channel, attention_encodings)
