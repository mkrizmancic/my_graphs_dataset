from pathlib import Path
from typing import Any, Union

import networkx as nx
from tqdm import tqdm


class GraphDataset:
    def __init__(self, selection=None, graph_format="graph6"):
        self.selection = selection
        self.format = graph_format

        self.file_name_template = "graphs_{}.txt"

        self.raw_files_dir = Path(__file__).parents[1] / "data" / graph_format
        self.raw_file_names = self._make_raw_file_names()
        self.num_graphs = self._make_num_graphs()

    def _make_raw_file_names(self):
        if not self.selection:
            return [file.name for file in sorted(self.raw_files_dir.iterdir())]

        # TODO: Impelement size filtering
        if isinstance(self.selection, (list, dict)):
            return [
                file.name
                for file in sorted(self.raw_files_dir.iterdir())
                if GraphDataset.extract_graph_size(file.name) in self.selection
            ]

        raise ValueError("Selection must be a list, a dictionary, or None.")

    def _make_num_graphs(self):
        num_graphs = {}
        for file_name in self.raw_file_names:
            with open(self.raw_files_dir / file_name) as file:
                graph_size = GraphDataset.extract_graph_size(file_name)
                num_graphs_in_file = len(file.readlines())
                if isinstance(self.selection, dict) and self.selection[graph_size] > 0:
                    num_graphs[graph_size] = min(self.selection[graph_size], num_graphs_in_file)
                else:
                    num_graphs[graph_size] = num_graphs_in_file
        return num_graphs

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
            for file_name in files_w_progress:
                files_w_progress.set_description(f"Processing {file_name}")
                # Load the number of graphs specified in the selection.
                num_graphs_to_load = self.num_graphs[GraphDataset.extract_graph_size(file_name)]
                _batch_size = num_graphs_to_load if batch_size == "auto" else int(batch_size)
                # Open the file and process it.
                with open(self.raw_files_dir / file_name, "r") as file:
                    batch: list[nx.Graph | str] = []
                    total_graphs_from_file = 0
                    for line in tqdm(file, total=num_graphs_to_load):
                        if total_graphs_from_file >= num_graphs_to_load:
                            break

                        if batch_size == 1:
                            yield self.load_graph(line, raw)
                            total_graphs_from_file += 1
                        else:
                            batch.append(self.load_graph(line, raw))
                            if len(batch) >= min(_batch_size, num_graphs_to_load - total_graphs_from_file):
                                yield batch
                                total_graphs_from_file += len(batch)
                                batch = []
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

    @staticmethod
    def parse_graph6(description):
        return nx.from_graph6_bytes(bytes(description.strip(), "ascii"))

    @staticmethod
    def parse_edgelist(description):
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
