import math

from my_graphs_dataset import GraphDataset, GraphType


def test_loading_and_saving():
    print("====================================")
    print("TESTING LOADING AND SAVING")

    # Read and generate individual graphs, and then save them to disk in graph6 and edgelist format.
    selection = {
        3: -1,
        4: -1,
        GraphType.LINE: (1, range(10, 15 + 1)),
    }
    loader = GraphDataset(selection=selection, seed=1, graph_format="graph6")
    loader.save_graphs("my_graphs")
    loader.save_graphs("my_graphs", graph_format="edgelist")

    # Read the graph6 format and print it out.
    selection = {"my_graphs": -1}
    new_loader = GraphDataset(selection=selection, graph_format="graph6")
    lst = [G for G in new_loader.graphs(batch_size=1, raw=True)]
    for G in lst:
        print(G)

    # Read the edgelist format and print it out.
    newest_loader = GraphDataset(selection=selection, graph_format="edgelist")
    lst = [G for G in newest_loader.graphs(batch_size=1, raw=True)]
    for G in lst:
        print(G)

    print("====================================\n")


def test_seed_reproducability():
    print("====================================")
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
        loader1 = GraphDataset(selection=selection, seed=seed, graph_format="graph6", suppress_output=True, retries=20)
        lst1 = [G for G in loader1.graphs(batch_size=1, raw=True)]

        loader2 = GraphDataset(selection=selection, seed=seed, graph_format="graph6", suppress_output=True, retries=20)
        lst2 = [G for G in loader2.graphs(batch_size=1, raw=True)]

        assert len(lst1) == len(lst2)
        assert set(lst1) == set(lst2)
        print("Seed reproducability test passed.")

    seed = 123
    for i in range(10):
        print(f"{i=}, {seed=} | ", end="")
        do_test(selection, seed)
        seed = int(math.sqrt(seed) * 42 + 13)

    print("====================================\n")


def test_all_graphs_generated():
    print("====================================")
    print("TESTING ALL GRAPHS GENERATED")
    selection = {
        ## Random and generated graphs
        #   Type of the graph: (num. graphs for each size, [sizes] OR range(sizes))
        #   Completly random graphs
        GraphType.BARABASI_ALBERT:      (100, range(10, 15+1)),  # 600
        GraphType.ERDOS_RENYI:          (100, range(10, 15+1)),  # 600
        GraphType.WATTS_STROGATZ:       (100, range(10, 15+1)),  # 600
        GraphType.NEW_WATTS_STROGATZ:   (100, range(10, 15+1)),  # 600
        GraphType.STOCH_BLOCK:          (100, range(10, 15+1)),  # 600
        GraphType.REGULAR:              (100, range(10, 15+1)),  # 600
        GraphType.CATERPILLAR:          (100, range(10, 15+1)),  # 600
        GraphType.LOBSTER:              (100, range(10, 15+1)),  # 600
        GraphType.POWER_TREE:           (10, range(10, 15+1)),   # 60
        #   Random graphs with limited variability
        GraphType.FULL_K_TREE:  (100, range(10, 15+1)),  # 6+7+8+9+10+11 = 51
        GraphType.LOLLIPOP:     (100, range(10, 15+1)),  # 7+8+9+10+11+12 = 57
        GraphType.BARBELL:      (100, range(10, 15+1)),  # 3+3+4+4+5+5 = 24
        #   Unique families of graphs
        GraphType.GRID:         (1, range(10, 15+1)),  # 6
        GraphType.CAVEMAN:      (1, range(10, 15+1)),  # 6
        GraphType.LADDER:       (1, range(10, 15+1)),  # 6
        GraphType.LINE:         (1, range(10, 15+1)),  # 6
        GraphType.STAR:         (1, range(10, 15+1)),  # 6
        GraphType.CYCLE:        (1, range(10, 15+1)),  # 6
        GraphType.WHEEL:        (1, range(10, 15+1)),  # 6
        ## All isomorphic graph with N nodes from a file.
        #   N: num. graphs OR -1 for all graphs
        # 3: -1,
        # 4: 3,
    }  # TOTAL = 5034

    loader = GraphDataset(selection=selection, seed=1, graph_format="graph6", retries=50)
    lst = [G for G in loader.graphs(batch_size=1, raw=True)]
    if len(lst) == 5034:
        print("TEST OK")
    else:
        print(f"TEST FAILED ({len(lst)}/5574)")

    print("====================================\n")


def test_limited_graphs_batch_limit():
    print("====================================")
    print("TESTING LIMITED GRAPHS BATCH LIMIT")
    selection = {
        #   Random graphs with limited variability
        GraphType.LOLLIPOP:     (100, range(10, 15+1)),  # 7+8+9+10+11+12 = 57
    }

    loader = GraphDataset(selection=selection, seed=1, graph_format="graph6", retries=50)

    for i, batch in enumerate(loader.graphs(batch_size=4, raw=False)):
        print(f"{i}: Batch size={len(batch)} {[len(G) for G in batch]}")

    print("====================================\n")


if __name__ == '__main__':
    test_loading_and_saving()
    test_seed_reproducability()
    test_all_graphs_generated()
    test_limited_graphs_batch_limit()