from pathlib import Path
from typing import Any, Union
from warnings import warn

import networkx as nx
from tqdm import tqdm

from my_graphs_dataset.graph_generators import GraphType, generate_graph


class GraphDataset:
    def __init__(self, selection=None, graph_format="graph6", connected=True, random_scale=0.5, retries=10, seed=None):
        """
        Initialize the dataset loader.

        Args:
            selection: A dictionary of the form {graph_type: (num_graphs, [graph_sizes])} or a list of graph sizes.
            graph_format: Format of the graphs in the dataset. Supported formats are "graph6" and "edgelist".
            connected: Force graphs to be connected.
            random_scale: Modifier 0-1 for how well the random graphs are connected.
            retries: Number of retries to generate a connected graph.
            seed: Seed for the random number generator.
        """
        self.selection = selection
        self.format = graph_format

        self.file_name_template = "graphs_{}.txt"

        self.seed = seed
        self.retries = retries
        self.connected = connected
        self.random_scale = random_scale

        self.raw_files_dir = Path(__file__).parents[1] / "data" / graph_format
        self.raw_file_names = self._make_raw_file_names()
        self.num_graphs = self._make_num_graphs()

    def _make_raw_file_names(self):
        """Return the list of filenames that need to be processed."""
        if not self.selection:
            return [file.name for file in sorted(self.raw_files_dir.iterdir())]

        if isinstance(self.selection, (list, dict)):
            return [
                file.name
                for file in sorted(self.raw_files_dir.iterdir())
                if GraphDataset.extract_graph_size(file.name) in self.selection
            ]

        raise ValueError("Selection must be a list, a dictionary, or None.")

    def _make_num_graphs(self):
        """Determine the number of graphs to load from each file."""
        num_graphs = {}
        new_selection = {}
        for file_name in self.raw_file_names:
            with open(self.raw_files_dir / file_name) as file:
                graph_size = GraphDataset.extract_graph_size(file_name)
                # Total number of graphs in the file.
                num_graphs_in_file = len(file.readlines())
                # Load the specified number of graphs or all if the specified number is greater.
                if isinstance(self.selection, dict) and self.selection[graph_size] > 0:
                    num_graphs[graph_size] = min(self.selection[graph_size], num_graphs_in_file)
                # Otherwise, load all graphs from the file.
                else:
                    num_graphs[graph_size] = num_graphs_in_file
                    new_selection[graph_size] = num_graphs_in_file

        # Save the actual number of loaded graphs to the selection.
        if new_selection:
            self.selection = {} if self.selection is None else self.selection
            self.selection.update(new_selection)

        return num_graphs

    @property
    def hashable_selection(self):
        return {str(k): (v[0], list(v[1])) if isinstance(v, tuple) else v for k, v in self.selection.items()}

    def graphs(self, batch_size: Union[str, int]=1, raw=False) -> Any:
        """
        Yield graphs from the dataset.

        Args:
            batch_size:
                - "auto": Yields graphs in batches of the same size as the number of graphs in individual files.
                - 1: Yields graphs one by one.
                - != 1: Yields graphs in batches equal to or less than the specified size. Files containing the graphs
                    are read individually so the batch might be smaller than the specified size.
            raw: If True, yields raw graph descriptions as strings. If False, yields NetworkX graphs. Raw descriptions
                are useful for parallel processing.
        """
        with tqdm(self.raw_file_names) as files_w_progress:
            # Iterate over all available files.
            files_w_progress.set_description("Processing fixed graphs")
            for file_name in files_w_progress:
                # Load the number of graphs specified in the selection.
                num_graphs_to_load = self.num_graphs[GraphDataset.extract_graph_size(file_name)]
                _batch_size = num_graphs_to_load if batch_size == "auto" else int(batch_size)
                # Open the file and process it.
                # TODO: Shuffle the graphs in the file to prevent always loading the same graphs.
                #   When this is implemented, save the used seed for dataset name hashing.
                with open(self.raw_files_dir / file_name, "r") as file:
                    batch: list[nx.Graph | str] = []
                    total_graphs_from_file = 0
                    for line in tqdm(file, total=num_graphs_to_load, desc=file_name):
                        if total_graphs_from_file >= num_graphs_to_load:
                            break

                        if batch_size == 1:
                            # Yield one graph.
                            yield self.load_graph(line, raw)
                            total_graphs_from_file += 1
                        else:
                            # Graphs will be appended to the batch in each iteration of the loop.
                            # Total number of graphs is not yet updated, so the loop will not break.
                            batch.append(self.load_graph(line, raw))
                            # When we reach the desired batch size or the desired number of graphs, yield the batch.
                            if len(batch) >= min(_batch_size, num_graphs_to_load - total_graphs_from_file):
                                yield batch
                                total_graphs_from_file += len(batch)
                                batch = []
                    # We reached the end of the file. Yield the final partial batch if it is not empty.
                    if batch:
                        yield batch

        # There are no more graphs to be generated.
        if self.selection is None:
            return

        graph_generators = {k: v for k, v in self.selection.items() if isinstance(k, GraphType)}
        with tqdm(graph_generators.items()) as generators_w_progress:
            generators_w_progress.set_description("Generating random graphs")
            for graph_type, num_graphs in generators_w_progress:
                # Load the number of graphs specified in the selection.
                num_graphs_to_generate = sum(num_graphs[0] for _ in num_graphs[1])
                _batch_size = num_graphs_to_generate if batch_size == "auto" else int(batch_size)
                # Iterate over the specified number of graphs.
                with tqdm(total=num_graphs_to_generate, desc=graph_type.padded_value) as pbar:
                    batch: list[nx.Graph | str] = []
                    total_graphs_generated = 0
                    for graph_size in num_graphs[1]:
                        for _ in range(num_graphs[0]):
                            if total_graphs_generated >= num_graphs_to_generate:
                                break

                            if batch_size == 1:
                                # Yield one graph.
                                yield self.generate_graph(graph_type, graph_size, raw)
                                total_graphs_generated += 1
                            else:
                                # Graphs will be appended to the batch in each iteration of the loop.
                                # Total number of graphs is not yet updated, so the loop will not break.
                                batch.append(self.generate_graph(graph_type, graph_size, raw))
                                # When we reach the desired batch size or the desired number of graphs, yield the batch.
                                if len(batch) >= min(_batch_size, num_graphs_to_generate - total_graphs_generated):
                                    yield batch
                                    total_graphs_generated += len(batch)
                                    batch = []

                            pbar.update(1)

                    if batch:
                        yield batch


    def load_graph(self, description:str, raw=False):
        if raw:
            return description

        if self.format == "graph6":
            return GraphDataset.parse_graph6(description)
        elif self.format == "edgelist":
            return GraphDataset.parse_edgelist(description)
        else:
            raise ValueError(f"Unknown graph format: {self.format}")

    def generate_graph(self, graph_type: GraphType, N: int, raw=False):
        """
        Generate a graph of the specified type.

        Args:
            graph_type: Type of the graph to generate.
            N: Number of nodes in the graph.
            raw: If True, returns the graph description as a string. If False, returns a NetworkX
        """
        G = generate_graph(N, graph_type, scale=self.random_scale, seed=self.seed)

        # Ensure that the graph is connected.
        retry_count = 0
        while self.connected and not nx.is_connected(G):
            if retry_count and retry_count % self.retries == 0:
                warn(f"Failed to generate a connected {graph_type.name} graph after {retry_count} retries.")
            if retry_count >= self.retries * 3:
                raise RuntimeError(f"Failed to generate a connected {graph_type.name} graph after {retry_count} retries.")
            G = generate_graph(N, graph_type, scale=self.random_scale, seed=self.seed)
            retry_count += 1

        if raw:
            return nx.to_graph6_bytes(G).decode("ascii")
        else:
            return G

    @staticmethod
    def parse_graph6(description):
        return nx.from_graph6_bytes(bytes(description.strip(), "ascii"))

    @staticmethod
    def parse_edgelist(description):
        # We assume that the index of the nodes is the same as the node label.
        # By default, networkx adds the nodes in the order they are found in the
        # edgelist. For example, if the edgelist is [(1, 3), (2, 3)], the order
        # of the nodes will be [1, 3, 2]. This interferes with the adjacency
        # matrix ordering. So, we first add sorted nodes and then the edges.
        edges = [tuple(map(int, edge.split())) for edge in description.split("; ")]
        G_init = nx.from_edgelist(edges)
        G = nx.Graph()
        G.add_nodes_from(sorted(G_init.nodes))
        G.add_edges_from(G_init.edges)
        return G

    @staticmethod
    def extract_graph_size(file_name):
        # graphs_05.txt -> 5
        return int(file_name.split(".")[0].split("_")[1])


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    selection = {
        ## Random and generated graphs
        #   Type of the graph: (num. graphs for each size, [sizes] OR range(sizes))
        GraphType.ERDOS_RENYI: (10, [10, 15, 20]),
        GraphType.BARABASI_ALBERT: (10, range(10, 15+1)),
        GraphType.GRID: (10, range(10, 15+1, 5)),
        GraphType.RANDOM_MIX: (10, [10, 15, 20]),
        ## All isomorphic graph with N nodes from a file.
        #   N: num. graphs OR -1 for all graphs
        3: -1,
        4: 3,
    }

    loader = GraphDataset(selection=selection)
    lst = []
    for G in loader.graphs(batch_size="auto"):
        lst.append(len(G))
        # nx.draw(G, with_labels=True)
        # plt.show()
    print(lst)
