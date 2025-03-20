import random
import re
from pathlib import Path
from typing import Any, Union
from warnings import warn

import yaml
import networkx as nx
import numpy as np
from tqdm import tqdm

from my_graphs_dataset.graph_generators import GraphType, generate_graph


class GraphDataset:
    def __init__(self, selection=None, graph_format="graph6", connected=True, random_scale=1.0, retries=10, seed=None, suppress_output=False):
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
        self.selection = selection if selection is not None else dict()
        self.format = graph_format

        self.file_name_template = "graphs_{}.txt"

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.retries = retries
        self.connected = connected
        self.random_scale = random_scale
        self.suppress_output = suppress_output

        self.seen_graphs = dict()

        self.raw_files_dir_base = Path(__file__).parents[1] / "data"
        self.raw_files_dir = self.raw_files_dir_base / graph_format
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
                if self.extract_graph_info(file.name) in self.selection
            ]

        raise ValueError("Selection must be a list, a dictionary, or None.")

    def _make_num_graphs(self):
        """Determine the number of graphs to load from each file."""
        num_graphs = {}
        new_selection = {}
        for file_name in self.raw_file_names:
            with open(self.raw_files_dir / file_name) as file:
                graph_size = self.extract_graph_info(file_name)
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

    def _prepare_seen_graphs(self, graph_type, N):
        if graph_type not in self.seen_graphs:
            self.seen_graphs[graph_type] = dict()

        if N not in self.seen_graphs[graph_type]:
            self.seen_graphs[graph_type][N] = set()

    @property
    def hashable_selection(self):
        return {str(k): (v[0], list(v[1])) if isinstance(v, tuple) else v for k, v in self.selection.items()}

    # TODO: It woul be great to generate graphs in parallel, but this might interfere with pytorch.
    def graphs(self, batch_size: Union[str, int] = 1, raw: str | bool = "auto") -> Any:
        """
        Yield graphs from the dataset.

        Args:
            batch_size:
                - "auto": Yields graphs in batches of the same size as the number of graphs in individual files.
                - 1: Yields graphs one by one.
                - != 1: Yields graphs in batches equal to or less than the specified size. Files containing the graphs
                    are read individually so the batch might be smaller than the specified size.
            raw:
                - True: yields raw graph descriptions as strings. Raw descriptions are useful for parallel processing.
                - False: yields NetworkX graphs.
                - "auto": yields raw descriptions for graphs read from files and NetworkX graphs for generated graphs.
        """
        # Reset the list of seen graphs for each call to this generator.
        self.seen_graphs = dict()

        with tqdm(self.raw_file_names, disable=self.suppress_output) as files_w_progress:
            # Iterate over all available files.
            files_w_progress.set_description("Processing fixed graphs")
            for file_name in files_w_progress:
                # Load the number of graphs specified in the selection.
                num_graphs_to_load = self.num_graphs[self.extract_graph_info(file_name)]
                _batch_size = num_graphs_to_load if batch_size == "auto" else int(batch_size)
                # Open the file and process it.
                # TODO: Shuffle the graphs in the file to prevent always loading the same graphs.
                #   When this is implemented, save the used seed for dataset name hashing.
                with open(self.raw_files_dir / file_name, "r") as file:
                    batch: list[nx.Graph | str] = []
                    total_graphs_from_file = 0
                    for line in tqdm(file, total=num_graphs_to_load, desc=file_name, disable=self.suppress_output):
                        line = line.strip()
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
        with tqdm(graph_generators.items(), disable=self.suppress_output) as generators_w_progress:
            generators_w_progress.set_description("Generating random graphs")
            for graph_type, num_graphs in generators_w_progress:
                # Load the number of graphs specified in the selection.
                num_graphs_to_generate = sum(num_graphs[0] for _ in num_graphs[1])
                _batch_size = num_graphs_to_generate if batch_size == "auto" else int(batch_size)
                # Iterate over the specified number of graphs.
                with tqdm(total=num_graphs_to_generate, desc=graph_type.padded_value, disable=self.suppress_output) as pbar:
                    batch: list[nx.Graph | str] = []
                    total_graphs_generated = 0
                    for graph_size in num_graphs[1]:
                        size_graphs_generated = 0
                        for _ in range(num_graphs[0]):
                            if total_graphs_generated >= num_graphs_to_generate:
                                break
                            if size_graphs_generated >= graph_type.max_graphs:
                                break

                            if batch_size == 1:
                                # Yield one graph.
                                yield self.generate_graph(graph_type, graph_size, raw)
                                total_graphs_generated += 1
                                size_graphs_generated += 1
                            else:
                                # Graphs will be appended to the batch in each iteration of the loop.
                                # Total number of graphs is not yet updated, so the loop will not break.
                                batch.append(self.generate_graph(graph_type, graph_size, raw))
                                # When we reach the desired batch size or the desired number of graphs, yield the batch.
                                if len(batch) >= min(_batch_size, num_graphs_to_generate - total_graphs_generated, graph_type.max_graphs, graph_type.max_graphs - size_graphs_generated):
                                    yield batch
                                    total_graphs_generated += len(batch)
                                    size_graphs_generated += len(batch)
                                    batch = []

                            pbar.update(1)

                    if batch:
                        yield batch


    def load_graph(self, description: str, raw: str | bool = True):
        if raw == "auto" or raw is True:
            return description
        elif raw is False:
            if self.format == "graph6":
                return GraphDataset.parse_graph6(description)
            elif self.format == "edgelist":
                return GraphDataset.parse_edgelist(description)
            else:
                raise ValueError(f"Unknown graph format: {self.format}")
        else:
            raise ValueError(f"Unknown value for raw: {raw}")

    def generate_graph(self, graph_type: GraphType, N: int, raw: str | bool = False):
        """
        Generate a graph of the specified type.

        Args:
            graph_type: Type of the graph to generate.
            N: Number of nodes in the graph.
            raw: If True, returns the graph description as a string. If False, returns a NetworkX
        """
        G = generate_graph(N, graph_type, scale=self.random_scale)

        # Ensure that the graph is connected and not already generated.
        retry_count = 0
        self._prepare_seen_graphs(graph_type, N)
        graph6 = GraphDataset.to_graph6(G)
        while (self.connected and not nx.is_connected(G)) or graph6 in self.seen_graphs[graph_type][N]:
            if retry_count and retry_count % self.retries == 0:
                warn(f"Failed to generate a new/connected {graph_type.name} graph with {N} nodes after {retry_count} retries.")
            if retry_count >= self.retries * 3:
                raise RuntimeError(
                    f"Failed to generate a new/connected {graph_type.name} graph with {N} nodes after {retry_count} retries."
                )
            G = generate_graph(N, graph_type, scale=self.random_scale)
            graph6 = GraphDataset.to_graph6(G)
            retry_count += 1

        self.seen_graphs[graph_type][N].add(graph6)

        if raw is True:
            return graph6 if self.format == "graph6" else GraphDataset.to_edgelist(G)
        elif raw == "auto" or raw is False:
            return G
        else:
            raise ValueError(f"Unknown value for raw: {raw}")

    def save_graphs(self, name, graph_format="auto", save_description=False):
        if graph_format == "auto":
            graph_format = self.format

        graphs_to_save = []
        for G in self.graphs(batch_size=1, raw="auto"):
            graphs_to_save.append(G)

        file_name = self.file_name_template.format(name)
        with open(self.raw_files_dir_base / graph_format / file_name, "w") as file:
            for G in graphs_to_save:
                if isinstance(G, nx.Graph):
                    description = GraphDataset.to_graph6(G) if graph_format == "graph6" else GraphDataset.to_edgelist(G)
                elif graph_format != self.format:
                    # The format in which the graphs are saved is different from the format in which they are loaded.
                    # We need to convert the graph to the desired format.
                    G = GraphDataset.parse_graph6(G) if self.format == "graph6" else GraphDataset.parse_edgelist(G)
                    description = GraphDataset.to_graph6(G) if graph_format == "graph6" else GraphDataset.to_edgelist(G)
                else:
                    description = G
                file.write(description + "\n")

        if save_description:
            self.save_generation_result(name, graph_format)

    def save_generation_result(self, name, graph_format):
        """Store the number and type of generated graphs for future reference."""
        # Calculate the total number of graphs for each size. This is the combination of loaded and generated graphs.
        per_size = dict()
        for key in self.selection:
            if isinstance(key, int):
                per_size[key] = self.selection[key]
        for graph_type in self.seen_graphs:
            for N in self.seen_graphs[graph_type]:
                per_size[N] = per_size.get(N, 0) + len(self.seen_graphs[graph_type][N])

        # Calculate the total number of graphs for each type.
        per_type = dict()
        for key in self.selection:
            if isinstance(key, int):
                per_type["PREDEFINED_UNIQUE"] = per_type.get("PREDEFINED_UNIQUE", 0) + self.selection[key]
            else:
                per_type[key.name] = sum(len(graphs) for graphs in self.seen_graphs[key].values())

        # Prepare the list of generated graphs per type and size.
        combined = dict()
        if self.selection is not None:
            combined["PREDEFINED_UNIQUE"] = {k: v for k, v in self.selection.items() if isinstance(k, int)}
        for graph_type in self.seen_graphs:
            combined[graph_type.name] = {N: len(graphs) for N, graphs in self.seen_graphs[graph_type].items()}

        # Write the results to a file.
        with open(self.raw_files_dir_base / graph_format / f"description_{name}.yaml", "w") as file:
            result = {"total": sum(per_size.values()), "per_size": per_size, "per_type": per_type, "combined": combined}
            yaml.safe_dump(result, file, sort_keys=False)


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
    def to_graph6(G):
        return nx.to_graph6_bytes(G).decode("ascii").split("<<")[1].strip()

    @staticmethod
    def to_edgelist(G):
        return "; ".join(nx.generate_edgelist(G, data=False))

    def extract_graph_info(self, file_name):
        # graphs_05.txt -> 5
        # graphs_my_graphs.txt -> my_graphs
        pattern = re.escape(self.file_name_template).replace("\\{\\}", "(.*)")
        match = re.match(pattern, file_name)
        if match is None:
            print(f"Skipping file {file_name}")
            return None

        try:
            info = int(match.group(1))
        except ValueError:
            info = match.group(1)
        return info


def join_descriptions(descriptions, output):
    total = 0
    per_size = dict()
    per_type = dict()
    combined = dict()

    for description in descriptions:
        with open(description, 'r') as file:
            data = yaml.safe_load(file)
            total += data["total"]
            per_size.update(data["per_size"])

            for key, value in data["per_type"].items():
                per_type[key] = per_type.get(key, 0) + value

            for key, value in data["combined"].items():
                combined[key] = combined.get(key, dict())
                combined[key].update(value)

    with open(output, 'w') as file:
        result = {"total": total, "per_size": per_size, "per_type": per_type, "combined": combined}
        yaml.safe_dump(result, file, sort_keys=False)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    selection = {
        ## Random and generated graphs
        #   Type of the graph: (num. graphs for each size, [sizes] OR range(sizes))
        #   Completly random graphs
        GraphType.BARABASI_ALBERT:      (100, range(26, 29+1, 1)),
        GraphType.ERDOS_RENYI:          (100, range(26, 29+1, 1)),
        GraphType.WATTS_STROGATZ:       (100, range(26, 29+1, 1)),
        GraphType.NEW_WATTS_STROGATZ:   (100, range(26, 29+1, 1)),
        GraphType.STOCH_BLOCK:          (100, range(26, 29+1, 1)),
        GraphType.REGULAR:              (100, range(26, 29+1, 1)),
        GraphType.CATERPILLAR:          (100, range(26, 29+1, 1)),
        GraphType.LOBSTER:              (100, range(26, 29+1, 1)),
        GraphType.POWER_TREE:           (100, range(26, 29+1, 1)),  # Must be a small number. There aren't that many power
                                                                # law trees. Check https://oeis.org/A000055/list.
                                                                # The number must be much smaller than in the list.
        #   Random graphs with limited variability
        GraphType.FULL_K_TREE:  (100, range(26, 29+1, 1)),
        GraphType.LOLLIPOP:     (100, range(26, 29+1, 1)),
        GraphType.BARBELL:      (100, range(26, 29+1, 1)),
        #   Unique families of graphs
        GraphType.GRID:         (1, range(26, 29+1, 1)),
        GraphType.CAVEMAN:      (1, range(26, 29+1, 1)),
        GraphType.LADDER:       (1, range(26, 29+1, 1)),
        GraphType.LINE:         (1, range(26, 29+1, 1)),
        GraphType.STAR:         (1, range(26, 29+1, 1)),
        GraphType.CYCLE:        (1, range(26, 29+1, 1)),
        GraphType.WHEEL:        (1, range(26, 29+1, 1)),
        ## All isomorphic graph with N nodes from a file.
        #   N: num. graphs OR -1 for all graphs
        # 3: -1,
        # 4: -1,
        # 5: -1,
        # 6: -1,
        # 7: -1,
        # 8: -1,
    }

    selection = {"26-50_mix_100": -1}

    loader = GraphDataset(selection=selection, seed=42, graph_format="graph6", retries=100)

    # lst = [G for G in loader.graphs(batch_size=1, raw=True)]
    # print(len(lst))

    # shit_counter = 0
    # for G in loader.graphs(batch_size=1, raw=False):
    #     if not nx.is_connected(G):
    #         shit_counter += 1
    # print(shit_counter)

    # loader.save_graphs("26-50_mix_100", graph_format="graph6", save_description=False)

    # descriptions = [loader.raw_files_dir_base / "graph6" / f"description_{name}.yaml" for name in selection]
    # join_descriptions(descriptions, "description_26-50_mix_100.yaml")

