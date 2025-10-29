# Clustered Echo State Network (CESN) for the prediciton of multivariate time series
This repository contains an implementation of Echo State Networks with various clustered topologies to predict multivariate time series.
In this, we have "cesn.py" containing classes of all ESNs including conventional ESN and clustered ESNs.

Clustered ESN:
1) Ring
2) Erdos Renyi
3) Scale-Free network

Different topologies are explored with sepcific input mapping to clusters and seperate readouts. The number of clusters in the
reservoir is decided by the number of variables to be predicted. Here, solar wind multivariate time series is taken as task and 
example usage is based on the same with 'example.ipynb' demonstrating the working of CESN. 

## Files
- `cesn.py` — Contains ESN class definitions:
  - `EchoStateNetwork`
  - `EchoStateNetworkWithBAClusters`
  - `EchoStateNetworkWithErdosRenyiClusters`
  - `EchoStateNetworkWithAdaptiveClusters`
- `utils_func.py` — Utility functions (e.g., data splitting, NRMSE computation)
- `example.ipynb` — Demonstration of using CESN with sample data
- `data/` — Folder containing example datasets



