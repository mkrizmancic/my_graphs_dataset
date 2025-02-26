"""
Generates random graphs of different types of a given size.
Some of the graph are created using the NetworkX library, for more info see
https://networkx.github.io/documentation/networkx-1.10/reference/generators.html
"""

import math
import random
from enum import Enum
from functools import partial
from typing import Counter

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def empty_graph(N, scale, seed):
    raise (ValueError("Random Type Graphs should not be called as a function."))


def erdos_renyi(N, scale, seed):
    """
    Generate an Erdős-Rényi or binomial graph of size `N` with probability of edge creation proportional to `scale`.

    If `scale == 1`, graphs of any density can be generated.
    If `scale < 1`, probability of edge creation is limited by `scale`.

    Maximum number of graphs = unlimited
    """
    return nx.fast_gnp_random_graph(N, random.random() * scale, seed, directed=False)


def barabasi_albert(N, scale, seed):
    """
    Generate a Barabási-Albert graph of size `N` and where nodes are attached with `degree` edges.

    Depending on `scale`, the `degree` is a random number in the range [0, N-1].

    Maximum number of graphs = unlimited
    """
    degree = random.randint(1, round((N - 1) * scale))
    return nx.barabasi_albert_graph(N, degree, seed)


def watts_strogatz(N, scale, seed):
    """
    Generate a small-world graph of size `N` with each node connected to its `k` nearest neighbors in a ring topology
    and then rewiring the edges with probability `p`.

    Depending on `scale`, `k` is a random number in the range [0, N-1] and `p` is a random number in the range [0, 1].

    Maximum number of graphs = unlimited
    """
    k = random.randint(2, round((N - 1) * scale))
    p = random.random() * scale
    return nx.connected_watts_strogatz_graph(N, k, p, tries=10000, seed=seed)


def newman_watts_strogatz(N, scale, seed):
    """
    Generate a small-world graph of size `N` with each node connected to its `k` nearest neighbors in a ring topology
    and then rewiring the edges with probability `p`. During rewiring, no edges are removed.

    Depending on `scale`, `k` is a random number in the range [0, N-1] and `p` is a random number in the range [0, 1].

    Maximum number of graphs = unlimited
    """
    k = round(random.random() * (N - 1) * scale)
    p = random.random() * scale
    return nx.newman_watts_strogatz_graph(N, k, p, seed)


def stochastic_block(N, scale, seed):
    """
    Generate a stochastic block model graph of size `N` with each node belonging to one of `n_blocks` blocks and edges
    created with probability `p` inside the same block and `q` between different blocks.

    Maximum number of graphs = unlimited
    """
    n_blocks = random.randint(1, round(N * scale))
    sizes = [N // n_blocks] * n_blocks
    for i in range(N % n_blocks):
        sizes[i] += 1

    p = random.random() * scale
    q = random.random() * scale
    probs = (np.ones((n_blocks, n_blocks)) - np.eye(n_blocks)) * q + np.eye(n_blocks) * p
    probs = probs.tolist()

    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    return G


def grid(N, scale, seed):
    """
    Generate a `m*k` 2D grid graph with `N = m*k` and `m` and `k` as close as possible.

    `m` and `k` could be made random, but (Corso et al., 2020) defines it like this.

    Maximum number of graphs = 1
    """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    assert m * (N // m) == N
    return nx.grid_2d_graph(m, N // m)


def ladder(N, scale, seed):
    """
    Generate a ladder graph of `N` nodes: two rows of N/2 nodes, with each pair connected by a single edge.

    In case `N` is odd another node is attached to the first one.

    Maximum number of graphs = 1
    """
    G = nx.ladder_graph(N // 2)
    if N % 2 != 0:
        G.add_node(N - 1)
        G.add_edge(0, N - 1)
    return G


def regular(N, scale, seed):
    """
    Generate a regular graph of size `N` with each node connected to `d` neighbors.

    `d` must be smaller than `N`, and `d*N` must be even.
    If `scale == 1`, select `d` as a random number between 1 or 2 and `N` exclusive.
    If `scale < 1`, it controls the maximum value of `d` as `round(N*scale)`.

    Example:
        - N = 10, scale = 1 -> d = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        - N = 10, scale = 0.5 -> d = [1, 2, 3, 4]
        - N = 11, scale = 1 -> d = [2, 4, 6, 8, 10]
        - N = 11, scale = 0.5 -> d = [2, 4]

    Maximum number of graphs = unlimited
    """
    if N % 2 == 0:
        # N is even - d can be even or odd.
        d = random.choice(range(1, round(N * scale), 1))
    else:
        # N is odd - d must be even.
        d = random.choice(range(2, round(N * scale), 2))

    return nx.random_regular_graph(d, N)


def line(N, scale, seed):
    """
    Generate a graph composed of `N` nodes in a line.

    Maximum number of graphs = 1
    """
    return nx.path_graph(N)


def star(N, scale, seed):
    """
    Generate a graph composed by one center node connected `N-1` outer nodes.

    Maximum number of graphs = 1
    """
    return nx.star_graph(N - 1)


def cycle(N, scale, seed):
    """
    Generate a graph composed of `N` nodes in a cycle.

    Maximum number of graphs = 1
    """
    return nx.cycle_graph(N)


def power_tree(N, scale, seed):
    """
    Generate a tree of size `N` with a power law degree distribution.

    Maximum number of graphs = unlimited?
    """
    return nx.random_powerlaw_tree(N, gamma=3, tries=10000)


def full_k_tree(N, scale, seed):
    """
    Generate a tree of size `N` with a k-ary distribution.

    For `k >= N - 2`, the graph is a star.

    Maximum number of graphs = round((N - 3) * scale) - 1
    """
    full_k_tree.max_graphs = round((N - 3) * scale)
    k = random.randint(2, round((N - 3) * scale))
    return nx.full_rary_tree(k, N)


def wheel(N, scale, seed):
    """
    Generate a wheel graph of size N with a center node connected to all the others.

    Maximum number of graphs = 1
    """
    return nx.wheel_graph(N)


def caveman(N, scale, seed):
    """
    Generate a caveman graph of `m` cliques of size `k`, with `m` and `k` as close as possible.

    `m` and `k` could be made random, but (Corso et al., 2020) defines it like this.

    Maximum number of graphs = 1
    """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    assert m * (N // m) == N
    return nx.connected_caveman_graph(m, N // m)


def caterpillar(N, scale, seed):
    """
    Generate a random caterpillar graph with a backbone of size `b` (drawn from U[1, N)),
    and `N-b` pendent vertices uniformly connected to the backbone.

    Maximum number of graphs = unlimited
    """
    B = np.random.randint(low=1, high=N)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, N):
        G.add_edge(i, np.random.randint(B))
    return G


def lobster(N, scale, seed):
    """
    Generate a random Lobster graph with a backbone of size `b` (drawn from U[1, N)),
    `p` (drawn from U[1, N - b ]) pendent vertices uniformly connected to the backbone,
    and additional `N-b-p` pendent vertices uniformly connected to the previous pendent vertices.

    Maximum number of graphs = unlimited
    """
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


def lolipop(N, scale, seed):
    """
    Generate a random lollipop graph with a complete graph of size `m` and a path of size `n`.

    Follows the definition on https://mathworld.wolfram.com/LollipopGraph.html.

    Maximum number of graphs = N - 3
    """
    lolipop.max_graphs = N - 3
    m = random.randint(3, N - 1)
    n = N - m
    G = nx.lollipop_graph(m, n)
    return G


def barbell(N, scale, seed):
    """
    Generate a random barbell graph with two complete graphs of size `m` and a path of size `n`.

    Follows the definition on https://mathworld.wolfram.com/BarbellGraph.html.

    Maximum number of graphs = N // 2 - 2
    """
    barbell.max_graphs = N // 2 - 2
    m = random.randint(3, N // 2)
    n = N - 2 * m
    G = nx.barbell_graph(m, n)
    return G


def randomize(A):
    """Adds some randomness by toggling some edges without changing the expected number of edges of the graph"""
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


class GraphType(Enum):
    RANDOM_PNA = partial(empty_graph)
    RANDOM_OUR = partial(empty_graph)
    ERDOS_RENYI = partial(erdos_renyi)
    BARABASI_ALBERT = partial(barabasi_albert)
    WATTS_STROGATZ = partial(watts_strogatz)
    NEW_WATTS_STROGATZ = partial(newman_watts_strogatz)
    STOCH_BLOCK = partial(stochastic_block)
    GRID = partial(grid)
    LADDER = partial(ladder)
    REGULAR = partial(regular)
    LINE = partial(line)
    STAR = partial(star)
    CYCLE = partial(cycle)
    POWER_TREE = partial(power_tree)
    FULL_K_TREE = partial(full_k_tree)
    WHEEL = partial(wheel)
    CAVEMAN = partial(caveman)
    CATERPILLAR = partial(caterpillar)
    LOBSTER = partial(lobster)
    LOLLIPOP = partial(lolipop)
    BARBELL = partial(barbell)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

    @property
    def padded_value(self):
        # Calculate the length of the longest enum value
        max_length = max(len(item.name.lower()) for item in GraphType)
        # Return the value with padding spaces to align all values
        return self.name.rjust(max_length).lower()


# probabilities of each type in case of random type
MIXTURE_PNA = [(GraphType.ERDOS_RENYI, 0.2), (GraphType.BARABASI_ALBERT, 0.2), (GraphType.GRID, 0.05),
               (GraphType.CAVEMAN, 0.05), (GraphType.POWER_TREE, 0.15), (GraphType.LADDER, 0.05),
               (GraphType.LINE, 0.05), (GraphType.STAR, 0.05), (GraphType.CATERPILLAR, 0.1), (GraphType.LOBSTER, 0.1)]

MIXTURE_OUR = [(GraphType.BARABASI_ALBERT, 0.2), (GraphType.ERDOS_RENYI, 0.2), (GraphType.WATTS_STROGATZ, 0.1),
               (GraphType.NEW_WATTS_STROGATZ, 0.1), (GraphType.STOCH_BLOCK, 0.05), (GraphType.GRID, 0.05),
               (GraphType.CAVEMAN, 0.03), (GraphType.POWER_TREE, 0.015), (GraphType.FULL_K_TREE, 0.015),
               (GraphType.LADDER, 0.03), (GraphType.LINE, 0.0125), (GraphType.STAR, 0.0125),
               (GraphType.CATERPILLAR, 0.03), (GraphType.LOBSTER, 0.03), (GraphType.REGULAR, 0.05),
               (GraphType.LOLLIPOP, 0.025), (GraphType.CYCLE, 0.0125), (GraphType.WHEEL, 0.0125),
               (GraphType.BARBELL, 0.025)]


def generate_graph(N, graph_type=GraphType.RANDOM_OUR, scale=0.5):
    """
    Generates random graphs of different types of a given size. Note:
     - graph are undirected and without weights on edges

    :param N:       number of nodes
    :param type:    type chosen between the categories specified in GraphType enum

    """
    # sample which random type to use
    if graph_type == GraphType.RANDOM_OUR:
        graph_type = np.random.choice([t for (t, _) in MIXTURE_OUR], 1, p=[pr for (_, pr) in MIXTURE_OUR])[0]

    try:
        G = graph_type(N, scale, seed=None)
        if len(G) == 0:
            raise ValueError("Empty graph generated")
    except Exception as e:
        print(f"Error generating graph {graph_type.name} for {N=} {scale=}.")
        raise e

    return G


def generate_and_analyze(N, graph_type, scale, seed):
    G = generate_graph(N, graph_type, scale)
    min_edges = N - 1
    max_edges = N * (N - 1) / 2
    density = (G.number_of_edges() - min_edges) / (max_edges - min_edges)
    avg_degree = sum(d for (_, d) in list(G.degree)) / N

    return nx.to_graph6_bytes(G).decode("ascii"), density, avg_degree


def plot_histograms(data, title, xlim):
    num_graphs = len(data)
    rows = math.ceil(math.sqrt(num_graphs))  # Arrange in a square-like grid
    cols = math.ceil(num_graphs / rows)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(data.keys()))

    for i, (key, values) in enumerate(data.items()):
        row = (i // cols) + 1
        col = (i % cols) + 1
        fig.add_trace(go.Histogram(x=values, name=key), row=row, col=col)

    fig.update_xaxes(range=xlim)
    fig.update_layout(
        title=title,
        showlegend=False,
    )
    fig.update_traces(xbins_size=xlim[1] / 10)

    fig.show()


def test_graph_generation(scale, seed):
    sn = 10
    en = 41

    # Define the types of graphs to generate.
    unique_types = [GraphType.GRID, GraphType.LADDER, GraphType.LINE, GraphType.STAR, GraphType.CYCLE, GraphType.WHEEL,
                    GraphType.CAVEMAN]
    limited_types = [GraphType.FULL_K_TREE, GraphType.LOLLIPOP, GraphType.BARBELL]
    random_types = [GraphType.ERDOS_RENYI, GraphType.BARABASI_ALBERT, GraphType.WATTS_STROGATZ,
                    GraphType.NEW_WATTS_STROGATZ, GraphType.STOCH_BLOCK, GraphType.REGULAR, GraphType.CATERPILLAR,
                    GraphType.LOBSTER,  GraphType.POWER_TREE]
    all_types = unique_types + limited_types + random_types

    # Prepare the data structures to store the generated graphs and their properties.
    all_graphs = {t.name: [] for t in (all_types)}
    densities = {t.name: [] for t in (all_types)}
    avg_degrees = {t.name: [] for t in (all_types)}
    by_sizes_unique = Counter()
    by_sizes_expected = Counter()

    # Define the function to iterate over the types of graphs and generate them.
    def iterate_over_types(graph_types, n):
        for t in graph_types:
            print(t.name)
            for N in range(sn, en, 2):
                graphs = set()
                for _ in range(n):
                    G, density, avg_degree = generate_and_analyze(N, t, scale, seed)
                    if G not in graphs:
                        graphs.add(G)
                        densities[t.name].append(density)
                        avg_degrees[t.name].append(avg_degree)
                num_unique = len(graphs)
                num_expected = t.value.func.max_graphs if hasattr(t.value.func, "max_graphs") else n
                by_sizes_unique[N] += num_unique
                by_sizes_expected[N] += num_expected
                print(f"  {N}: {num_unique}/{num_expected} ({num_unique / num_expected * 100:.2f}%)")
                all_graphs[t.name].extend(graphs)

    # Generate the graphs.
    n = 100
    iterate_over_types(unique_types, 1)
    iterate_over_types(limited_types, n)
    iterate_over_types(random_types, n)

    # Print the percentage of unique graphs generated.
    print(
        f"\nNumber of unique graphs for {scale=}. "
        f"Total average: {sum(by_sizes_unique.values()) / sum(by_sizes_expected.values()) * 100:.2f}"
    )
    for N in by_sizes_unique:
        print(
            f"  {N}: {by_sizes_unique[N]}/{by_sizes_expected[N]} ({by_sizes_unique[N] / by_sizes_expected[N] * 100:.2f}%)"
        )

    # Plot distributions of densities and average degrees.
    plot_histograms(densities, f"Density distribution - {scale=}", (0, 1))
    plot_histograms(avg_degrees, f"Average degree distribution - {scale=}", (0, en))

    return all_graphs


if __name__ == "__main__":
    scale = 0.5
    seed = 123

    random.seed(seed)
    np.random.seed(seed)

    graphs1 = test_graph_generation(scale, seed=None)

    # random.seed(seed)
    # np.random.seed(seed)

    # graphs2 = test_graph_generation(scale, seed=None)

    # # Are the graphs the same?
    # for t in graphs1.keys():
    #     if graphs1[t] != graphs2[t]:
    #         print(f"Graphs for {t} are different")
    #         print(graphs1[t])
    #         print(graphs2[t])
