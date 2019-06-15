"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 The matrix is converted to bias vectors.
"""
def adj_to_bias(adj):
    return -1e9 * (1.0 - adj)
