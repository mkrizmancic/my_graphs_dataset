import numpy as np
import random
import networkx as nx
import math
import matplotlib.pyplot as plt  # only required to plot
from enum import Enum


"""
    Generates random graphs of different types of a given size.
    Some of the graph are created using the NetworkX library, for more info see
    https://networkx.github.io/documentation/networkx-1.10/reference/generators.html
"""


class GraphType(Enum):
    RANDOM_MIX = "random_mix"
    ERDOS_RENYI = "erdos_renyi"
    BARABASI_ALBERT = "barabasi_albert"
    GRID = "grid"
    CAVEMAN = "caveman"
    TREE = "tree"
    LADDER = "ladder"
    LINE = "line"
    STAR = "star"
    CATERPILLAR = "caterpillar"
    LOBSTER = "lobster"

    @property
    def padded_value(self):
        # Calculate the length of the longest enum value
        max_length = max(len(item.value) for item in GraphType)
        # Return the value with padding spaces to align all values
        return self.value.rjust(max_length)


# probabilities of each type in case of random type
MIXTURE = [(GraphType.ERDOS_RENYI, 0.2), (GraphType.BARABASI_ALBERT, 0.2), (GraphType.GRID, 0.05),
           (GraphType.CAVEMAN, 0.05), (GraphType.TREE, 0.15), (GraphType.LADDER, 0.05),
           (GraphType.LINE, 0.05), (GraphType.STAR, 0.05), (GraphType.CATERPILLAR, 0.1), (GraphType.LOBSTER, 0.1)]


def erdos_renyi(N, scale, seed):
    """ Creates an Erdős-Rényi or binomial graph of size N with probability of edge creation proportional scale"""
    return nx.fast_gnp_random_graph(N, random.random() * scale, seed, directed=False)


def barabasi_albert(N, scale, seed):
    """ Creates a random graph according to the Barabási–Albert preferential attachment model
        of size N and where nodes are atteched with degree edges """
    degree = int(random.random() * (N - 1) * scale) + 1
    if degree == N:
        degree = N - 1
    return nx.barabasi_albert_graph(N, degree, seed)


def grid(N):
    """ Creates a m x k 2d grid graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.grid_2d_graph(m, N // m)


def caveman(N):
    """ Creates a caveman graph of m cliques of size k, with m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.caveman_graph(m, N // m)


def tree(N, seed):
    """ Creates a tree of size N with a power law degree distribution """
    return nx.random_powerlaw_tree(N, seed=seed, tries=10000)


def ladder(N):
    """ Creates a ladder graph of N nodes: two rows of N/2 nodes, with each pair connected by a single edge.
        In case N is odd another node is attached to the first one. """
    G = nx.ladder_graph(N // 2)
    if N % 2 != 0:
        G.add_node(N - 1)
        G.add_edge(0, N - 1)
    return G


def line(N):
    """ Creates a graph composed of N nodes in a line """
    return nx.path_graph(N)


def star(N):
    """ Creates a graph composed by one center node connected N-1 outer nodes """
    return nx.star_graph(N - 1)


def caterpillar(N, seed):
    """ Creates a random caterpillar graph with a backbone of size b (drawn from U[1, N)), and N − b
        pendent vertices uniformly connected to the backbone. """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, N):
        G.add_edge(i, np.random.randint(B))
    return G


def lobster(N, seed):
    """ Creates a random Lobster graph with a backbone of size b (drawn from U[1, N)), and p (drawn
        from U[1, N − b ]) pendent vertices uniformly connected to the backbone, and additional
        N − b − p pendent vertices uniformly connected to the previous pendent vertices """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    F = np.random.randint(low=B + 1, high=N + 1)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, F):
        G.add_edge(i, np.random.randint(B))
    for i in range(F, N):
        G.add_edge(i, np.random.randint(low=B, high=F))
    return G


def randomize(A):
    """ Adds some randomness by toggling some edges without changing the expected number of edges of the graph """
    BASE_P = 0.9

    # e is the number of edges, r the number of missing edges
    N = A.shape[0]
    e = np.sum(A) / 2
    r = N * (N - 1) / 2 - e

    # ep chance of an existing edge to remain, rp chance of another edge to appear
    if e <= r:
        ep = BASE_P
        rp = (1 - BASE_P) * e / r
    else:
        ep = BASE_P + (1 - BASE_P) * (e - r) / e
        rp = 1 - BASE_P

    array = np.random.uniform(size=(N, N), low=0.0, high=0.5)
    array = array + array.transpose()
    remaining = np.multiply(np.where(array < ep, 1, 0), A)
    appearing = np.multiply(np.multiply(np.where(array < rp, 1, 0), 1 - A), 1 - np.eye(N))
    ans = np.add(remaining, appearing)

    # assert (np.all(np.multiply(ans, np.eye(N)) == np.zeros((N, N))))
    # assert (np.all(ans >= 0))
    # assert (np.all(ans <= 1))
    # assert (np.all(ans == ans.transpose()))
    return ans


def generate_graph(N, graph_type=GraphType.RANDOM_MIX, seed=None, scale=0.5):
    """
    Generates random graphs of different types of a given size. Note:
     - graph are undirected and without weights on edges
     - node values are sampled independently from U[0,1]

    :param N:       number of nodes
    :param type:    type chosen between the categories specified in GraphType enum
    :param seed:    random seed
    :param degree:  average degree of a node, only used in some graph types
    :return:        adj_matrix: N*N numpy matrix
                    node_values: numpy array of size N
    """
    random.seed(seed)
    np.random.seed(seed)

    # sample which random type to use
    if graph_type == GraphType.RANDOM_MIX:
        graph_type = np.random.choice([t for (t, _) in MIXTURE], 1, p=[pr for (_, pr) in MIXTURE])[0]

    # generate the graph structure depending on the type
    if graph_type == GraphType.ERDOS_RENYI:
        G = erdos_renyi(N, scale, seed)
    elif graph_type == GraphType.BARABASI_ALBERT:
        G = barabasi_albert(N, scale, seed)
    elif graph_type == GraphType.GRID:
        G = grid(N)
    elif graph_type == GraphType.CAVEMAN:
        G = caveman(N)
    elif graph_type == GraphType.TREE:
        G = tree(N, seed)
    elif graph_type == GraphType.LADDER:
        G = ladder(N)
    elif graph_type == GraphType.LINE:
        G = line(N)
    elif graph_type == GraphType.STAR:
        G = star(N)
    elif graph_type == GraphType.CATERPILLAR:
        G = caterpillar(N, seed)
    elif graph_type == GraphType.LOBSTER:
        G = lobster(N, seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return G


if __name__ == '__main__':
    for i in range(100):
        adj_matrix, node_values = generate_graph(10, GraphType.RANDOM, seed=i)
    print(adj_matrix)