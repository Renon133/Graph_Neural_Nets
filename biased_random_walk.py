import random
import networkx as nx
import numpy as np

def biased_random_walk(G, node_start, p, q, length):
    """
    Perform a biased random walk on a graph.

    Parameters:
    G (networkx.Graph): The input graph.
    node_start (int): The starting node for the random walk.
    p (float): Return parameter. Controls the likelihood of returning to the previous node.
    q (float): In-out parameter. Controls the likelihood of visiting nodes further away.
    length (int): The length of the random walk.

    Returns:
    list: A list of nodes representing the random walk.
    """
    walk = [node_start]  # Initialize the walk with the starting node
    while len(walk) < length:
        current_node = walk[-1]  # Get the current node
        neighbors = list(G.neighbors(current_node))  # Get the neighbors of the current node
        if len(neighbors) > 0:
            if len(walk) == 1:
                # If this is the first step, choose a random neighbor
                walk.append(random.choice(neighbors))
            else:
                previous_node = walk[-2]  # Get the previous node in the walk
                next_node = random.choice(neighbors)  # Choose a random neighbor
                if next_node == previous_node:
                    # If the next node is the previous node, append it
                    walk.append(next_node)
                elif G.has_edge(previous_node, next_node):
                    # If there is an edge between the previous node and the next node, append it
                    walk.append(next_node)
                else:
                    # Otherwise, append the previous node
                    walk.append(previous_node)
        else:
            # If there are no neighbors, break the loop
            break
    return walk