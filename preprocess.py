import numpy as np


def adjacency_matrix_to_edges(A: np.ndarray[np.ndarray]) -> np.ndarray:

    n = A.shape[0]
    source_nodes = []
    target_nodes = []
    edge_weights = []

    for j in range(n):
        for i in range(n):
            if A[i][j] != 0:
                source_nodes.append(i)
                target_nodes.append(j)
                edge_weights.append(A[i][j])
    
    edge_index = [source_nodes, target_nodes]
    edge_attr = edge_weights

    return edge_index, edge_attr
