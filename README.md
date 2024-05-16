# My Graphs Dataset
I am working on various research about graphs so I often need a datest of different graphs to test my algorithms. I am aware that there are plenty of datasets available online but I wanted to have a custom solution I can adapt to my needs and I also wanted practice my Python skills.

## Contents
### Graphs
At the moment, the dataset consists of the following graphs:
1. All unique undirected graphs with 3 - 10 vertices:

    | Vertices     | 3 | 4 | 5 | 6 | 7 |  8  |   9  |   10   |
    |------------- |---|---|---|---|---|-----|------|--------|
    | No. of graphs| 2 | 6 | 21|112|853|11117|261080|11716571|

### Scripts
1. `converter.py` - Converts the graphs between different formats.
   1. grap6 <--> edgelist
   1. ...
1. `file_list_snapshot.py` - Creates a snapshot of the file list in the dataset.
1. `dataset_loader.py` - Loads the dataset into memory.


## Usage
### Generating the dataset
The files are not actually stored because they can be easily generated. I am using the great `geng` tool from the `nauty` package to generate the graphs. 
> McKay, B.D. and Piperno, A., Practical Graph Isomorphism, II,
Journal of Symbolic Computation, 60 (2014), pp. 94-112, https://doi.org/10.1016/j.jsc.2013.09.003  
> 
> Available at https://pallini.di.uniroma1.it/

The precompiled version of the `geng` tool is available from the Releases section. Download the executable into the main `my_graphs_dataset` directory and run it with
```bash
for i in {03..10}; do ./geng -cl $i > graph6/graphs_$i.txt; done
```
This will generate all unique graphs with 3 to 10 vertices in the graph6 format. The graphs are stored in the `graph6` directory.

> [!TIP]
> The graphs are stored in graph6 format because it is the most compact format for storing graphs. The graphs can be easily converted to other formats using the `converter.py` script.

### Installing the package
Clone the repository and install the package with pip. You may use the optional `-e` flag to install the package in the editable mode.
```bash
git clone git@github.com:mkrizmancic/my_graphs_dataset.git
cd my_graphs_dataset
pip install [-e] -r requirements.txt .
```

### Examples of using the dataset
#### Exploring graphs
```python
import networkx as nx
from matplotlib import pyplot as plt

from my_graphs_dataset import GraphDataset

loader = GraphDataset()

for G in loader.graphs():  # G is a networkx.Graph
    print(nx.to_numpy_array(G))
    nx.draw(G, with_labels=True)
    plt.show()
```

#### Loading a selection of graphs
```python
from my_graphs_dataset import GraphDataset

# Load all available graphs.
loader = GraphDataset(selection=None)
loader = GraphDataset(selection=[])

# Load all graphs with 3, 4, and 5 vertices.
loader = GraphDataset(selection=[3, 4, 5])  
# Load all graphs with 3 and 4 vertices and first 10 graphs with 5 vertices.
loader = GraphDataset(selection={3: -1, 4: -1, 5: 10})  
```

#### Parallel processing
```python
from tqdm.contrib.concurrent import process_map
from my_graphs_dataset import GraphDataset

def worker(graph):
    G = GraphDataset.parse_graph6(graph)
    # Do something with the graph
    return res

loader = GraphDataset()
all_results = []

# If batch_size="auto", loader yields all graphs from individual files.
for graphs in loader.graphs(raw=True, batch_size=10000):
    # Process map runs the multiprocessing pool and displays a progress bar with tqdm.
    result = process_map(worker, graphs, chunksize=1000)
    all_results.extend(result)
```