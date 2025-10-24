import math

from my_graphs_dataset import GraphDataset, GraphType


def test_loading_and_saving():
    print("==========================")
    print("TESTING LOADING AND SAVING")

    # Read and generate individual graphs, and then save them to disk in graph6 and edgelist format.
    selection = {
        3: -1,
        4: -1,
        GraphType.LINE: (1, range(10, 15 + 1)),
    }
    loader = GraphDataset(selection=selection, seed=1, graph_format="graph6", suppress_warnings=True)
    loader.save_graphs("my_graphs")
    loader.save_graphs("my_graphs", graph_format="edgelist")

    # Read the graph6 format and print it out.
    selection = {"my_graphs": -1}
    new_loader = GraphDataset(selection=selection, graph_format="graph6", suppress_warnings=True)
    lst = [G for G in new_loader.graphs(batch_size=1, raw=True)]
    for G in lst:
        print(G)

    # Read the edgelist format and print it out.
    newest_loader = GraphDataset(selection=selection, graph_format="edgelist", suppress_warnings=True)
    lst = [G for G in newest_loader.graphs(batch_size=1, raw=True)]
    for G in lst:
        print(G)

    print("==========================\n")


def test_seed_reproducability():
    print("============================")
    print("TESTING SEED REPRODUCABILITY")
    selection = {
        ## Random and generated graphs
        #   Type of the graph: (num. graphs for each size, [sizes] OR range(sizes))
        #   Completly random graphs
        GraphType.BARABASI_ALBERT:      (10, range(10, 15+1)),
        GraphType.ERDOS_RENYI:          (10, range(10, 15+1)),
        GraphType.WATTS_STROGATZ:       (10, range(10, 15+1)),
        GraphType.NEW_WATTS_STROGATZ:   (10, range(10, 15+1)),
        GraphType.STOCH_BLOCK:          (10, range(10, 15+1)),
        GraphType.REGULAR:              (10, range(10, 15+1)),
        GraphType.CATERPILLAR:          (10, range(10, 15+1)),
        GraphType.LOBSTER:              (10, range(10, 15+1)),
        GraphType.POWER_TREE:           (10, range(10, 15+1)),
        #   Random graphs with limited variability
        GraphType.FULL_K_TREE:  (10, range(10, 15+1)),
        GraphType.LOLLIPOP:     (10, range(10, 15+1)),
        GraphType.BARBELL:      (10, range(10, 15+1)),
    }

    def do_test(selection, seed):
        loader1 = GraphDataset(
            selection=selection,
            seed=seed,
            graph_format="graph6",
            suppress_output=True,
            retries=20,
            suppress_warnings=True,
        )
        lst1 = [G for G in loader1.graphs(batch_size=1, raw=True)]

        loader2 = GraphDataset(
            selection=selection,
            seed=seed,
            graph_format="graph6",
            suppress_output=True,
            retries=20,
            suppress_warnings=True,
        )
        lst2 = [G for G in loader2.graphs(batch_size=1, raw=True)]

        assert len(lst1) == len(lst2)
        assert set(lst1) == set(lst2)
        print("Seed reproducability test passed.")

    seed = 123
    for i in range(10):
        print(f"{i=}, {seed=} | ", end="")
        do_test(selection, seed)
        seed = int(math.sqrt(seed) * 42 + 13)

    print("============================\n")


def test_limited_graphs_batch_limit():
    print("===================================")
    print("TESTING LIMITED GRAPHS BATCH LIMIT")
    selection = {
        #   Random graphs with limited variability
        GraphType.LOLLIPOP: (100, range(10, 15 + 1)),  # 7+8+9+10+11+12 = 57
    }

    loader = GraphDataset(selection=selection, seed=1, graph_format="graph6", retries=50, suppress_warnings=True)

    for i, batch in enumerate(loader.graphs(batch_size=4, raw=False)):
        print(f"{i}: Batch size={len(batch)} {[len(G) for G in batch]}")

    print("===================================\n")


def test_uniqueness():
    from collections import Counter

    print("============================")
    print("TESTING UNIQUENESS OF GRAPHS")
    selection = {
        # "09_mix_1000": -1,
        # "10_mix_1000": -1,
        # "11-15_mix_200": -1,
        # "16-20_mix_200": -1,
        # "21-25_mix_200": -1,
        # "50_mix_200": -1
        "test": -1
    }
    loader = GraphDataset(
        selection=selection, seed=42, graph_format="graph6", retries=100, suppress_output=True, suppress_warnings=True
    )

    total = Counter()
    unique = Counter()
    seen = set()

    for graphs in loader.graphs(batch_size=10000, raw=True):
        canonical_labels = GraphDataset.canonical_label(graphs)

        for g, cl in zip(graphs, canonical_labels):
            size = GraphDataset.size_from_graph6(g)
            total[size] += 1
            if cl not in seen:
                unique[size] += 1
            seen.add(cl)

    header = False
    for size in sorted(total.keys()):
        if total[size] != unique[size]:
            if not header:
                print("Size\tTotal\tUnique\tPercentage")
                header = True
            print(f"{size}\t{total[size]}\t{unique[size]}\t{unique[size] / total[size] * 100:.2f}")
    if not header:
        print("OK! All graphs are unique!")
    print("============================")


if __name__ == "__main__":
    test_loading_and_saving()
    test_seed_reproducability()
    test_limited_graphs_batch_limit()
    test_uniqueness()
