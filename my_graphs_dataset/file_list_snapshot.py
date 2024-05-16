import argparse
from pathlib import Path

import yaml


def save_directory_structure(files_path, output_file):
    def traverse(path):
        structure = {}
        files = []
        path = Path(path)
        what_return = None
        for item in path.iterdir():
            if item.is_dir():
                structure[item.name] = traverse(item)
                what_return = structure
            else:
                files.append(item.name)
                what_return = files

        return what_return

    structure = traverse(files_path)
    with open(output_file, "w") as file:
        yaml.dump(structure, file)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--dir", type=str, default="graph6", help="Files directory to snapshot.")
    arg.add_argument("--output", type=str, default="file_list.yaml", help="Output file name.")

    args = arg.parse_args()

    files_path = Path(__file__).parents[1] / "data" / args.dir
    output_file = Path().cwd() / args.output
    save_directory_structure(files_path, output_file)
