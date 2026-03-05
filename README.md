# JTEPA-2026-bus-network-topology-ridership-pattern

This repository contains the data and code supporting the paper:

**"Investigating the Relationship Between Bus Network Topology and Temporal Ridership Patterns: A Case Study in Singapore"**
by [Wei Zhou](https://orcid.org/0000-0002-3608-1047), accepted for publication in the *[Journal of Transportation Engineering, Part A: Systems](https://ascelibrary.org/journal/jtepbs)*.

* Published paper (DOI): [10.1061/JTEPBS/TEENG-9665](https://doi.org/10.1061/JTEPBS/TEENG-9665)
* Archive of the accepted manuscript: [`manuscript/manuscript.pdf`](manuscript/manuscript.pdf)

---

## Paper

### Abstract

To advance the understanding of passenger temporal ridership profiles in urban bus systems, this study examines the relationship between network topological characteristics and temporal ridership patterns. Using Singapore’s bus system as a case study, stop-level hourly passenger profiles are constructed, and a clustering approach is applied to identify five typical temporal patterns: mixed-use central business district (CBD), western job hubs, peripheral residential regions, near-central residential regions, and school-oriented zones. Adopting the complex network theory, the bus network is represented in both L-space and P-space, allowing to compute key topological descriptors and to examine degree distributions and small-world features. A multinomial logit (MNL) model is employed to investigate the relationships between temporal ridership clusters and network topology. The results confirm significant associations and identify that residential origins exhibit stronger outward connectivity and bridging roles, whereas job and school destinations demonstrate concentrated inflows and high local transitivity within dense route cliques. Additionally, incorporating network topology significantly improves model performance, highlighting that network context provides additional explanatory power for temporal ridership patterns beyond the urban environment alone. By linking network topology with temporal ridership dynamics, this study provides a network-based perspective that can inform more efficient and adaptive public bus network planning.

**Keywords:** urban bus system, temporal ridership pattern, clustering analysis, network topology, multinomial logit model


## Getting started

### Project path configuration

Before running the notebooks, set the project root directory in the constant variable `PROJECT_PATH` in:

* [`code_utils/utils_basic.py`](code_utils/utils_basic.py)

This path is used throughout the notebooks to locate input data and write outputs consistently.

### Environment

Typical dependencies include: `numpy`, `scipy`, `pandas`, `geopandas`, `networkx`, `scikit-learn`, `statsmodels`, `matplotlib`, `scienceplots`.


## Data

This repository uses bus network information and bus stop ridership for October 2022 in Singapore. The datasets are organized into original and processed outputs.

### Original data (from LTA DataMall)

The raw files are retrieved from the **Singapore Land Transport Authority (LTA) DataMall**, a public platform providing transportation-related datasets.

* Source: Singapore **LTA DataMall**: [https://datamall.lta.gov.sg/](https://datamall.lta.gov.sg/)
* Archive:

  * [`bus_info_202210.zip`](data/bus_network/bus_info_202210.zip)

    * `bus_route_20221020.csv`: bus route information, including bus route, stop sequence, stop ID, and stop name.
    * `bus_stop_20221020.csv`: bus stop attributes, including bus stop ID, name, and coordinates (latitude, longitude).
    * `transport_node_bus_passenger_volume_202210.csv`: *hourly* boarding and alighting volumes for a typical workday and weekend day.


## Methods and workflow

### 1) Construct bus network topology graphs

A bus network topology graph is created using the route stop-sequence information with two representations:

* **L-space representation**: stops are connected if they are consecutive along a route
* **P-space representation**: stops are connected if they are co-served by at least one route

**Notebook**

* [`create_bus_graph.ipynb`](experiment/create_bus_graph.ipynb)

**Outputs**

* [`bus_graph_20221020.zip`](data/bus_network/bus_graph_20221020.zip)
  * `edge_list_space_l.csv`
  * `edge_list_space_p.csv`
  * `graph_space_l.graphml`
  * `graph_space_p.graphml`


### 2) Compute topological indicators

Topological indicators are computed for bus stops, including:

* In-/Out-degree centrality
* Closeness centrality
* Betweenness centrality
* In-/Out-Eigenvector centrality
* In-/Out-Kats centrality
* Cluster coefficient
* PageRank 

**Notebook**

* [`topological_metrics.ipynb`](experiment/topological_metrics.ipynb): compute topological indicators
* [`plot_degree_distribution.ipynb`](experiment/plot_data/plot_degree_distribution.ipynb): explores degree distributions of the topology graphs

**Outputs**

* [`features.zip`](data/features.zip)
    * `space_l_node_topology.csv`
    * `space_p_node_topology.csv`


### 3) Process temporal bus stop ridership

Extract *boarding* (tap-in) volumes for *workday* only within window to *06:00–23:00*.

**Notebook**

* [`process_passenger_volume.ipynb`](experiment/process_passenger_volume.ipynb)
* [`plot_volume_spatial_distribution.ipynb`](experiment/plot_data/plot_volume_spatial_distribution.ipynb): visualize spatial distribution of bus stop ridership

**Outputs**

* [`features.zip`](data/features.zip)

  * `busstop_volume_workday.csv`


### 4) Cluster temporal ridership patterns

Cluster bus stops based on temporal ridership profiles (hourly boarding patterns) using:

* K-means clustering
* Hierarchical clustering with Ward, average, complete, and single linkages

**Notebook**

* [`clustering.ipynb`](experiment/clustering.ipynb)
* [`plot_cluster_result.ipynb`](experiment/plot_data/plot_cluster_result.ipynb)

**Outputs**

* [`results.zip`](results/results.zip)

  * `clustering/clustering_criteria.csv`
    Validity metrics across different cluster numbers and methods.
  * `clustering/clustering_centers.csv`
  * `clustering/clustering_results_kmeans.csv`


### 5) Multinomial logistic (MNL) model analysis

A multinomial logit (MNL) model is used to examine associations between ridership-pattern cluster membership and explanatory variables (e.g., topology indicators and land use features).

**Notebook**

* [`mnl_model.ipynb`](experiment/mnl_model.ipynb)

**Outputs**

* [`results.zip`](results/results.zip)

  * `results_mnlogit_undersample.xlsx`

---

## Repository structure

```text
.
├─ code_utils/
│  ├─ utils_basic.py                         # project root path configuration (PROJECT_PATH)
│  └─ *.py                                   # utility modules used by notebooks
├─ data/
│  ├─ bus_network/
│  │  ├─ bus_graph_20221020.zip              # generated topology graphs, including L-space and P-space
│  │  │  ├─ edge_list_space_l.csv            # edge list for L-space graph, derived from bus route stop sequence
│  │  │  ├─ edge_list_space_p.csv            # edge list for P-space graph
│  │  │  ├─ graph_space_l.graphml            # L-space graph in GraphML format, for network analysis
│  │  │  └─ graph_space_p.graphml            # P-space graph in GraphML format
│  │  └─ bus_info_202210.zip                 # raw data from LTA DataMall
│  │     ├─ bus_route_20221020.csv           # bus route stop sequence, e.g., stop order, stop ID, stop name)
│  │     ├─ bus_stop_20221020.csv            # stop attributes, eg., ID, name, lat/lon
│  │     └─ transport_node_bus_passenger_volume_202210.csv
│  │                                         # hourly boarding/alighting for workday/weekend)
│  ├─ dataset.zip                            # combined datasets for MNL, incl. undersampled variants
│  │  ├─ space_l.csv
│  │  ├─ space_l_undersample.csv
│  │  ├─ space_p.csv
│  │  └─ space_p_undersample.csv
│  └─ features.zip                           # extracted features used in clustering and MNL
│     ├─ busstop_landuse_buffer_100.csv      # land-use composition within 100 m buffer
│     ├─ busstop_landuse_buffer_200.csv      # 200 m buffer
│     ├─ busstop_landuse_buffer_400.csv      # 400 m buffer
│     ├─ busstop_volume_workday.csv          # weekday hourly boarding volume (06:00–23:00)
│     ├─ space_l_node_topology.csv           # stop-level topology indicators (L-space)
│     └─ space_p_node_topology.csv           # P-space
├─ experiment/
│  ├─ clustering.ipynb                       # cluster temporal ridership patterns
│  ├─ create_bus_graph.ipynb                 # build topology graphs from route-stop sequences
│  ├─ mnl_model.ipynb                        # multinomial logistic regression analysis
│  ├─ process_passenger_volume.ipynb         # clean/aggregate ridership time series
│  ├─ topological_metrics.ipynb              # compute stop-level topology indicators
│  └─ plot_data/                             # Data and results visualization
│     ├─ plot clustering results.ipynb       # clustering visualizations, e.g., centers, spatial distribution
│     ├─ plot_degree_distribution.ipynb      # degree distribution of topology graphs
│     ├─ plot_feature_correlation.ipynb      # feature correlation diagnostics for MNL
│     ├─ plot_land_use_composition.ipynb     # land-use composition visualization
│     └─ plot_volume_spatial_distribution.ipynb
│                                            # spatial distribution of bus stop ridership
└─ results/
   └─ results.zip                            # exported outputs (clustering + MNL results)
      ├─ results_mnlogit_undersample.xlsx    # MNL outputs for undersampled dataset
      └─ clustering/
         ├─ clustering_centers.csv           # cluster centers, i.e., mean temporal profiles
         ├─ clustering_criteria.csv          # validity metrics across clustering numbers and methods
         └─ clustering_results_kmeans.csv    # cluster membership labels
```

## Citation

If you find this repository useful for your research, please consider citing this work as:

```bibtex
@article{zhou_bus_network_topology_ridership_2026,
  author  = {Zhou, Wei},
  title   = {Investigating the Relationship Between Bus Network Topology and Temporal Ridership Patterns: A Case Study in Singapore},
  journal = {Journal of Transportation Engineering, Part A: Systems},
  year    = {Forthcoming},
  doi     = {10.1061/JTEPBS/TEENG-9665}
}
```


## License

This project is released under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.


> Data note: The original data are retrieved from the Singapore LTA DataMall and remain subject to [LTA's data usage policies](https://datamall.lta.gov.sg/content/datamall/en/term-of-use.html).


## Contact

For questions, please open a GitHub Issue: [GitHub Issue](https://github.com/betterweizh/JTEPA-2026-bus-network-topology-ridership-pattern/issues).
