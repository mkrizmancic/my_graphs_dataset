import math
from unittest import loader
from dataset_loader import GraphDataset
from graph_generators import GraphType

def test_loading_and_saving():
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


def test_seed_reproducability():
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
        loader1 = GraphDataset(selection=selection, seed=seed, graph_format="graph6", suppress_output=True)
        lst1 = [G for G in loader1.graphs(batch_size=1, raw=True)]

        loader2 = GraphDataset(selection=selection, seed=seed, graph_format="graph6", suppress_output=True)
        lst2 = [G for G in loader2.graphs(batch_size=1, raw=True)]

        assert len(lst1) == len(lst2)
        assert set(lst1) == set(lst2)
        print("Seed reproducability test passed.")

    seed = 123
    for i in range(10):
        print(f"{i=}, {seed=} | ", end="")
        do_test(selection, seed)
        seed = int(math.sqrt(seed) * 42 + 13)


if __name__ == '__main__':
    test_seed_reproducability()