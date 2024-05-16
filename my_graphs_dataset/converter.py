import argparse
from pathlib import Path

import networkx as nx
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from my_graphs_dataset import GraphDataset


def graph6_to_edgelist(graph6: str) -> str:
    graph = GraphDataset.parse_graph6(graph6)
    return "; ".join(nx.generate_edgelist(graph, data=False)) + "\n"


def edgelist_to_graph6(edgelist: str) -> str:
    graph = GraphDataset.parse_edgelist(edgelist)
    return nx.to_graph6_bytes(graph).decode("ascii") + "\n"


def main():
    # Convert from graph6 format to edgelist format.
    # Take the input and output file paths as arguments.
    arg = argparse.ArgumentParser()
    arg.add_argument("--input", type=str, default="graph6", help="Input files directory.")
    arg.add_argument("--output", type=str, default="edgelist", help="Output files directory.")
    args = arg.parse_args()

    input_dir = Path(__file__).parent / "data" / args.input
    output_dir = Path(__file__).parent / "data" / args.output
    output_dir.mkdir(exist_ok=True, parents=True)

    with tqdm(input_dir.iterdir()) as files_w_progress:
        for filename in files_w_progress:
            files_w_progress.set_description(f"Processing {filename.name}")
            with open(filename, "r") as in_file:
                input_graphs = in_file.readlines()

            result = process_map(graph6_to_edgelist, input_graphs, chunksize=1000)

            with open(output_dir / filename.name, "w") as out_file:
                out_file.writelines(result)

            files_w_progress.update()


if __name__ == "__main__":
    main()
