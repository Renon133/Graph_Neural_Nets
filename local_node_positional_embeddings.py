import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def compute_positional_encodings(graph, max_hop):
    """
    Computes local node positional encodings for a given graph.

    Parameters:
        graph (networkx.Graph): The input graph.
        max_hop (int): The maximum hop distance to consider for encoding.

    Returns:
        dict: A dictionary where keys are nodes and values are positional encoding vectors.
    """
    # Initialize a dictionary to store positional encodings
    positional_encodings = {}

    # Compute shortest path lengths between all pairs of nodes
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    # Iterate through each node in the graph
    for node in graph.nodes:
        # Initialize the positional encoding vector for the current node
        encoding = np.zeros(max_hop)

        # Get the shortest path distances from the current node to all other nodes
        distances = shortest_paths[node]

        # Update the positional encoding vector based on distances
        for target_node, distance in distances.items():
            if 1 <= distance <= max_hop:
                encoding[distance - 1] += 1

        # Store the encoding vector
        positional_encodings[node] = encoding

    return positional_encodings

def compute_edge_based_positional_encodings(graph, edge_features, embedding_dim):
    """
    Computes local node positional encodings using edge features.

    Parameters:
        graph (networkx.Graph): The input graph.
        edge_features (dict): A dictionary where keys are (node1, node2) tuples and values are edge features.
        embedding_dim (int): The dimension of the edge feature embeddings.

    Returns:
        dict: A dictionary where keys are nodes and values are positional encoding vectors.
    """
    # Initialize a dictionary to store the positional encodings
    positional_encodings = {}

    # Define a learnable function for edge feature embeddings (example: random initialization)
    def learnable_embedding(edge_feature):
        np.random.seed(hash(edge_feature) % (2**32))  # Ensure consistent embeddings for the same feature
        return np.random.rand(embedding_dim)

    # Iterate through each node in the graph
    for node in graph.nodes:
        # Initialize a list to store edge embeddings for the current node
        edge_embeddings = []

        # Iterate through neighbors of the node
        for neighbor in graph.neighbors(node):
            edge_key = (node, neighbor) if (node, neighbor) in edge_features else (neighbor, node)
            if edge_key in edge_features:
                edge_feature = edge_features[edge_key]
                edge_embedding = learnable_embedding(edge_feature)
                edge_embeddings.append(edge_embedding)

        # Aggregate edge embeddings (e.g., by averaging)
        if edge_embeddings:
            positional_encodings[node] = np.mean(edge_embeddings, axis=0)
        else:
            positional_encodings[node] = np.zeros(embedding_dim)

    return positional_encodings

def compute_kernel_based_positional_encodings(graph, kernel_function):
    """
    Computes local node positional encodings using graph kernels.

    Parameters:
        graph (networkx.Graph): The input graph.
        kernel_function (callable): A function that computes similarity between two subgraphs.

    Returns:
        dict: A dictionary where keys are nodes and values are positional encoding vectors.
    """
    positional_encodings = {}

    for node in graph.nodes:
        subgraph_i = nx.ego_graph(graph, node)  # Get the subgraph around the node
        encoding = []

        for neighbor in graph.neighbors(node):
            subgraph_j = nx.ego_graph(graph, neighbor)  # Subgraph around the neighbor
            similarity = kernel_function(subgraph_i, subgraph_j)  # Compute kernel similarity
            encoding.append(similarity)

        positional_encodings[node] = np.array(encoding)

    return positional_encodings

def visualize_graph(graph, encodings, title):
    """
    Visualizes the graph with node embeddings as labels.

    Parameters:
        graph (networkx.Graph): The input graph.
        encodings (dict): A dictionary of node embeddings.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)  # Layout for visualization
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')

    for node, (x, y) in pos.items():
        plt.text(x, y + 0.1, s=np.round(encodings[node], 2), bbox=dict(facecolor='white', alpha=0.5),
                 horizontalalignment='center', fontsize=8)

    plt.title(title)
    plt.show()

def compare_embeddings(graph, shortest_path_encodings, edge_based_encodings, kernel_based_encodings):
    """
    Compares different embedding strategies visually.

    Parameters:
        graph (networkx.Graph): The input graph.
        shortest_path_encodings (dict): Shortest path-based encodings.
        edge_based_encodings (dict): Edge feature-based encodings.
        kernel_based_encodings (dict): Kernel-based encodings.
    """
    visualize_graph(graph, shortest_path_encodings, "Shortest Path-Based Encodings")
    visualize_graph(graph, edge_based_encodings, "Edge Feature-Based Encodings")
    visualize_graph(graph, kernel_based_encodings, "Kernel-Based Encodings")

# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (2, 4)])

    # Define edge features
    edge_features = {
        (1, 2): "type_A",
        (2, 3): "type_B",
        (3, 4): "type_A",
        (4, 5): "type_C",
        (1, 5): "type_B",
        (2, 4): "type_C"
    }

    # Define the embedding dimension
    embedding_dim = 4

    # Compute shortest path-based encodings
    max_hop = 3
    shortest_path_encodings = compute_positional_encodings(G, max_hop)

    # Compute edge-based positional encodings
    edge_based_encodings = compute_edge_based_positional_encodings(G, edge_features, embedding_dim)

    # Define a dummy kernel function (example: size similarity between subgraphs)
    def dummy_kernel(subgraph1, subgraph2):
        return len(subgraph1.nodes) / len(subgraph2.nodes)

    # Compute kernel-based positional encodings
    kernel_based_encodings = compute_kernel_based_positional_encodings(G, dummy_kernel)

    # Compare embeddings
    compare_embeddings(G, shortest_path_encodings, edge_based_encodings, kernel_based_encodings)
