import numpy as np
import networkx as nx
from scipy.linalg import pinv



# Custom Echo State Network class
class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius, leak_rate,
                 input_scaling, rc_connectivity, input_connectivity, seed):
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        
        # Initialize input weights with [-1, 1] range and scaling
        self.Win = (np.random.rand(n_reservoir, n_inputs) - 0.5) * 2 * input_scaling
        mask_input = np.random.rand(n_reservoir, n_inputs) < input_connectivity
        self.Win = np.multiply(self.Win, mask_input)  # Apply input connectivity mask

        # Initialize recurrent weights with [-1, 1] range and sparsity
        W = (np.random.rand(n_reservoir, n_reservoir) - 0.5) * 2
        mask_rc = np.random.rand(n_reservoir, n_reservoir) < rc_connectivity
        W = np.multiply(W, mask_rc)  # Apply recurrent connectivity mask
        
        rhoW = max(abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / rhoW)

        # Output weights 
        self.Wout = np.zeros((n_outputs, n_reservoir))
        
        # Reservoir state initialization
        self.reservoir_state = np.zeros((n_reservoir,))
        
    def _update_reservoir(self, input_signal):
        #computation of preactivation and updation of reservoir state
        pre_activation = np.dot(self.Win, input_signal) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * np.tanh(pre_activation)
        
    def reset_reservoir(self):
        #resetting reservoir state to zero
        self.reservoir_state = np.zeros((self.n_reservoir,))

    
    def fit(self, inputs, outputs, ridge_lambda=0.0, reset=False,washout = 100):     
        #Train the ESN on input-output pairs.
        if reset:
            self.reset_reservoir()  # Reset reservoir state if specified
        
        states = np.zeros((len(inputs), self.n_reservoir))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            states[t] = self.reservoir_state

        states = states[washout:]
        outputs = outputs[washout:]

        # Train output weights using ridge regression
        if ridge_lambda > 0:
            I = np.identity(self.n_reservoir)
            self.Wout = np.dot(np.dot(outputs.T, states), 
                               np.linalg.inv(np.dot(states.T, states) + ridge_lambda * I))
        else:
            self.Wout = np.dot(outputs.T, pinv(states.T))

    def predict(self, inputs, reset=False,washout=100):
        """
        Predict outputs given input sequences.

        Parameters:
        - inputs: array-like, shape (timesteps, input_dim)
        - reset: bool, default False. If True, reset the reservoir state before prediction.

        Returns:
        - predictions: array, shape (timesteps, output_dim)
        """
        if reset:
            self.reset_reservoir()  # Reset reservoir state if specified
        
        predictions = np.zeros((len(inputs), self.n_outputs))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            predictions[t] = np.dot(self.Wout, self.reservoir_state)
        return predictions[washout:]
        
        
#CESN - SCALE-FREE NETWORK
class EchoStateNetworkWithBAClusters:
    def __init__(self, n_inputs, neurons_per_cluster, n_clusters=None, spectral_radius=0.9, leak_rate=1.0,
                 input_scaling=1.0, mean_degree=3, inter_cluster_connectivity=0.01,
                 input_connectivity=0.1, seed=None):
        np.random.seed(seed)

        if n_clusters is None:
            n_clusters = n_inputs

        self.n_inputs = n_inputs
        self.neurons_per_cluster = neurons_per_cluster
        self.n_clusters = n_clusters
        self.n_reservoir = n_clusters * neurons_per_cluster
        self.n_outputs = n_inputs

        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.mean_degree = mean_degree
        self.inter_cluster_connectivity = inter_cluster_connectivity

        self.Win = np.zeros((self.n_reservoir, self.n_inputs))
        self.W = np.zeros((self.n_reservoir, self.n_reservoir))

        self._initialize_weights(input_connectivity)

        self.Wout_clusters = [np.zeros((1, self.neurons_per_cluster)) for _ in range(self.n_clusters)]
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def _initialize_weights(self, input_connectivity):
        neurons_per_cluster = self.neurons_per_cluster
        clusters = [range(i * neurons_per_cluster, (i + 1) * neurons_per_cluster) for i in range(self.n_clusters)]
        G = nx.Graph()
        G.add_nodes_from(range(self.n_reservoir))

        def add_intra_cluster_edges(G, clusters, mean_degree):
            for cluster in clusters:
                m = max(1, mean_degree // 2)
                ba_graph = nx.barabasi_albert_graph(len(cluster), m)
                mapping = {i: cluster[i] for i in range(len(cluster))}
                nx.relabel_nodes(ba_graph, mapping, copy=False)
                G.add_edges_from(ba_graph.edges())

        def add_inter_cluster_edges(G, clusters, icc_prob):
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    for ni in clusters[i]:
                        for nj in clusters[j]:
                            if np.random.rand() < icc_prob:
                                G.add_edge(ni, nj)

        add_intra_cluster_edges(G, clusters, self.mean_degree)
        add_inter_cluster_edges(G, clusters, self.inter_cluster_connectivity)

        W_graph = nx.to_numpy_array(G)
        self.W = (np.random.rand(self.n_reservoir, self.n_reservoir) * 2 - 1) * W_graph

        rhoW = max(abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / rhoW

        for i in range(self.n_inputs):
            start = i * self.neurons_per_cluster
            end = (i + 1) * self.neurons_per_cluster
            cluster_weights = (np.random.rand(self.neurons_per_cluster) * 2 - 1) * self.input_scaling
            mask_input = np.random.rand(self.neurons_per_cluster) < input_connectivity
            self.Win[start:end, i] = cluster_weights * mask_input

    def _update_reservoir(self, input_signal):
        input_signal = input_signal.reshape(-1)
        pre_activation = np.dot(self.Win, input_signal) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * np.tanh(pre_activation)

    def reset_reservoir(self):
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def fit(self, inputs, outputs, ridge_lambda=0.0, reset=False, washout=100):
        if reset:
            self.reset_reservoir()

        n_steps = len(inputs)
        states = np.zeros((n_steps, self.n_reservoir))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            states[t] = self.reservoir_state

        states = states[washout:]
        outputs = outputs[washout:]

        for cluster_idx in range(self.n_clusters):
            start = cluster_idx * self.neurons_per_cluster
            end = (cluster_idx + 1) * self.neurons_per_cluster
            cluster_states = states[:, start:end]
            target = outputs[:, cluster_idx].reshape(-1, 1)

            if ridge_lambda > 0:
                I = np.identity(self.neurons_per_cluster)
                self.Wout_clusters[cluster_idx] = np.dot(
                    target.T,
                    np.dot(cluster_states, np.linalg.inv(
                        np.dot(cluster_states.T, cluster_states) + ridge_lambda * I
                    ))
                )
            else:
                self.Wout_clusters[cluster_idx] = np.dot(target.T, pinv(cluster_states.T))

    def predict(self, inputs, reset=False, washout=100):
        if reset:
            self.reset_reservoir()

        n_steps = len(inputs)
        predictions = np.zeros((n_steps, self.n_outputs))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            for cluster_idx in range(self.n_clusters):
                start = cluster_idx * self.neurons_per_cluster
                end = (cluster_idx + 1) * self.neurons_per_cluster
                cluster_state = self.reservoir_state[start:end]
                predictions[t, cluster_idx] = np.dot(self.Wout_clusters[cluster_idx], cluster_state).item()

        return predictions[washout:]




# CESN - Erdős-Rényi intra-cluster connections
class EchoStateNetworkWithErdosRenyiClusters:
    def __init__(self, n_inputs, neurons_per_cluster, n_clusters=None, spectral_radius=0.9, leak_rate=1.0,
                 input_scaling=1.0, p_in=0.1, p_out=0.01, input_connectivity=0.1, seed=None):
        np.random.seed(seed)
        if n_clusters is None:
            n_clusters = n_inputs

        self.n_inputs = n_inputs
        self.neurons_per_cluster = neurons_per_cluster
        self.n_clusters = n_clusters
        self.n_reservoir = n_clusters * neurons_per_cluster
        self.n_outputs = n_inputs  # Each cluster is responsible for one output

        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.p_in = p_in
        self.p_out = p_out

        # Initialize input weights and recurrent weights
        self.Win = np.zeros((self.n_reservoir, self.n_inputs))
        self.W = np.zeros((self.n_reservoir, self.n_reservoir))
        self.structure_matrix = np.zeros_like(self.W)  # Matrix to store the structure of clusters

        self._initialize_weights(input_connectivity, spectral_radius, input_scaling)

        # Instead of a single readout, we create one readout per cluster.
        # Each readout maps from neurons_per_cluster to 1 output.
        self.Wout_clusters = [np.zeros((1, self.neurons_per_cluster)) for _ in range(self.n_clusters)]
        
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def _initialize_weights(self, input_connectivity, spectral_radius, input_scaling):
        # Set up input weights to be cluster-specific
        for cluster_idx in range(self.n_clusters):
            cluster_start = cluster_idx * self.neurons_per_cluster
            cluster_end = (cluster_idx + 1) * self.neurons_per_cluster

            # Each input i (cluster_idx) connects only to its designated cluster neurons.
            self.Win[cluster_start:cluster_end, cluster_idx] = ((np.random.rand(self.neurons_per_cluster) - 0.5) * 2 
                                                                 * input_scaling)
            mask_input = np.random.rand(self.neurons_per_cluster) < input_connectivity
            self.Win[cluster_start:cluster_end, cluster_idx] = np.multiply(self.Win[cluster_start:cluster_end, cluster_idx],
                                                                            mask_input)

            # Create intra-cluster connections (Erdős-Rényi) for neurons within the same cluster.
            for i in range(cluster_start, cluster_end):
                for j in range(cluster_start, cluster_end):
                    if i != j and np.random.rand() < self.p_in:
                        self.W[i, j] = (np.random.rand() - 0.5) * 2
                        self.structure_matrix[i, j] = 1  # Mark as intra-cluster connection

        # Create inter-cluster connections in range [-1, 1]
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                cluster_i_start = i * self.neurons_per_cluster
                cluster_i_end = (i + 1) * self.neurons_per_cluster
                cluster_j_start = j * self.neurons_per_cluster
                cluster_j_end = (j + 1) * self.neurons_per_cluster

                for ni in range(cluster_i_start, cluster_i_end):
                    for nj in range(cluster_j_start, cluster_j_end):
                        if np.random.rand() < self.p_out:
                            self.W[ni, nj] = (np.random.rand() - 0.5) * 2
                            self.W[nj, ni] = (np.random.rand() - 0.5) * 2
                            self.structure_matrix[ni, nj] = 2  # Mark as inter-cluster connection
                            self.structure_matrix[nj, ni] = 2

        # Normalize recurrent weight matrix to have the specified spectral radius
        rhoW = max(abs(np.linalg.eigvals(self.W)))
        self.W *= spectral_radius / rhoW


    def _update_reservoir(self, input_signal):
        input_signal = input_signal.reshape(-1)  # Ensure input_signal is 1D
        pre_activation = np.dot(self.Win, input_signal) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * np.tanh(pre_activation)

    def reset_reservoir(self):
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def fit(self, inputs, outputs, ridge_lambda=0.0, reset=False,washout=100):
        if reset:
            self.reset_reservoir()
        
        n_steps = len(inputs)
        states = np.zeros((n_steps, self.n_reservoir))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            states[t] = self.reservoir_state

        states = states[washout:]
        outputs = outputs[washout:]

        # For each cluster, train the readout using only the states from that cluster.
        for cluster_idx in range(self.n_clusters):
            cluster_start = cluster_idx * self.neurons_per_cluster
            cluster_end = (cluster_idx + 1) * self.neurons_per_cluster
            cluster_states = states[:, cluster_start:cluster_end]  # (n_steps, neurons_per_cluster)
            target = outputs[:, cluster_idx].reshape(-1, 1)  # (n_steps, 1) for this cluster

            if ridge_lambda > 0:
                I = np.identity(self.neurons_per_cluster)
                self.Wout_clusters[cluster_idx] = np.dot(
                    target.T,
                    np.dot(cluster_states, np.linalg.inv(np.dot(cluster_states.T, cluster_states) + ridge_lambda * I))
                )
            else:
                self.Wout_clusters[cluster_idx] = np.dot(target.T, pinv(cluster_states.T))

    def predict(self, inputs, reset=False,washout=100):
        """
        Predict outputs given input sequences using separate readouts for each cluster.
        
        Parameters:
        - inputs: array-like, shape (timesteps, input_dim)
        - reset: bool, if True, reset the reservoir state before prediction.
        
        Returns:
        - predictions: array, shape (timesteps, output_dim)
        """
        if reset:
            self.reset_reservoir()
        
        n_steps = len(inputs)
        predictions = np.zeros((n_steps, self.n_outputs))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            for cluster_idx in range(self.n_clusters):
                cluster_start = cluster_idx * self.neurons_per_cluster
                cluster_end = (cluster_idx + 1) * self.neurons_per_cluster
                cluster_state = self.reservoir_state[cluster_start:cluster_end]
                # Extract the scalar value from the dot product
                pred_value = np.dot(self.Wout_clusters[cluster_idx], cluster_state).item()
                predictions[t, cluster_idx] = pred_value
            
        return predictions[washout:]



# RING CESN
class EchoStateNetworkWithAdaptiveClusters:
    def __init__(self, n_inputs, neurons_per_cluster, n_clusters=None, spectral_radius=0.9, leak_rate=1.0,
                 input_scaling=1.0, intra_cluster_connectivity=1.0, inter_cluster_connectivity=0.01,
                 input_connectivity=0.1, seed=None):
                 
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Automatically determine number of clusters from the number of inputs
        if n_clusters is None:
            n_clusters = n_inputs  # One cluster per input 

        self.n_inputs = n_inputs
        self.neurons_per_cluster = neurons_per_cluster
        self.n_clusters = n_clusters  
        self.n_reservoir = n_clusters * neurons_per_cluster  # Total number of reservoir nodes
        self.n_outputs = n_inputs  # Outputs 
        
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.intra_cluster_connectivity = intra_cluster_connectivity
        self.inter_cluster_connectivity = inter_cluster_connectivity

        # Initialize input weights (Win) and recurrent weights (W)
        self.Win = np.zeros((self.n_reservoir, self.n_inputs))
        self.W = np.zeros((self.n_reservoir, self.n_reservoir))

        # Create the reservoir with clusters
        self._initialize_weights(input_connectivity, spectral_radius, input_scaling)

        # Instead of one readout for the entire reservoir,
        # create a separate readout for each cluster.
        # Each readout maps from neurons_per_cluster to 1 output.
        self.Wout_clusters = [np.zeros((1, self.neurons_per_cluster)) for _ in range(self.n_clusters)]

        # Reservoir state initialized to zero
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def _initialize_weights(self, input_connectivity, spectral_radius, input_scaling):
        """
        Initialize the weights using a ring network model for intra-cluster connections
        and sparse connections for inter-cluster connections.
        Also, assign input weights so that each input only targets its designated cluster.
        """
        for cluster_idx in range(self.n_clusters):
            cluster_start = cluster_idx * self.neurons_per_cluster
            cluster_end = (cluster_idx + 1) * self.neurons_per_cluster

            # Assign input weights for each input cluster (1 cluster per input) in range [-1, 1]
            self.Win[cluster_start:cluster_end, cluster_idx] = (np.random.rand(self.neurons_per_cluster) * 2 - 1) * input_scaling
            mask_input = np.random.rand(self.neurons_per_cluster) < input_connectivity
            self.Win[cluster_start:cluster_end, cluster_idx] = np.multiply(self.Win[cluster_start:cluster_end, cluster_idx], mask_input)

            # Initialize recurrent connections within the cluster (ring network) in range [-1, 1]
            for i in range(cluster_start, cluster_end):
                next_neuron = cluster_start + (i - cluster_start + 1) % self.neurons_per_cluster
                prev_neuron = cluster_start + (i - cluster_start - 1) % self.neurons_per_cluster
                self.W[i, next_neuron] = np.random.rand() * 2 - 1  # Weight in [-1, 1]
                self.W[i, prev_neuron] = np.random.rand() * 2 - 1  # Weight in [-1, 1]

        # Add inter-cluster connections (with a lower probability) in range [-1, 1]
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                cluster_i_start = i * self.neurons_per_cluster
                cluster_i_end = (i + 1) * self.neurons_per_cluster
                cluster_j_start = j * self.neurons_per_cluster
                cluster_j_end = (j + 1) * self.neurons_per_cluster

                for ni in range(cluster_i_start, cluster_i_end):
                    for nj in range(cluster_j_start, cluster_j_end):
                        if np.random.rand() < self.inter_cluster_connectivity:
                            self.W[ni, nj] = np.random.rand() * 2 - 1  # Weight in [-1, 1]
                            self.W[nj, ni] = np.random.rand() * 2 - 1  # Weight in [-1, 1]

        # Normalize recurrent weight matrix to have the specified spectral radius
        rhoW = max(abs(np.linalg.eigvals(self.W)))
        self.W *= spectral_radius / rhoW

    def _update_reservoir(self, input_signal):
        input_signal = input_signal.reshape(-1)  # Ensure input_signal is 1D
        pre_activation = np.dot(self.Win, input_signal) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * np.tanh(pre_activation)

    def reset_reservoir(self):
        """Resets the reservoir state to zero."""
        self.reservoir_state = np.zeros((self.n_reservoir,))

    def fit(self, inputs, outputs, ridge_lambda=0.0, reset=False,washout=100):
        if reset:
            self.reset_reservoir()
        
        n_steps = len(inputs)
        # Collect reservoir states over time
        states = np.zeros((n_steps, self.n_reservoir))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            states[t] = self.reservoir_state

        
        states = states[washout:]
        outputs = outputs[washout:]

        # Train separate readouts for each cluster using only the states from that cluster.
        for cluster_idx in range(self.n_clusters):
            cluster_start = cluster_idx * self.neurons_per_cluster
            cluster_end = (cluster_idx + 1) * self.neurons_per_cluster
            # Extract states for the current cluster
            cluster_states = states[:, cluster_start:cluster_end]  # shape: (n_steps, neurons_per_cluster)
            # The target for this cluster is the corresponding output column
            target = outputs[:, cluster_idx].reshape(-1, 1)  

            if ridge_lambda > 0:
                I = np.identity(self.neurons_per_cluster)
                self.Wout_clusters[cluster_idx] = np.dot(
                    target.T,
                    np.dot(cluster_states, np.linalg.inv(np.dot(cluster_states.T, cluster_states) + ridge_lambda * I))
                )
            else:
                self.Wout_clusters[cluster_idx] = np.dot(target.T, pinv(cluster_states.T))

    def predict(self, inputs, reset=False,washout=100):
        if reset:
            self.reset_reservoir()
        
        n_steps = len(inputs)
        predictions = np.zeros((n_steps, self.n_outputs))
        for t, input_signal in enumerate(inputs):
            self._update_reservoir(input_signal)
            for cluster_idx in range(self.n_clusters):
                cluster_start = cluster_idx * self.neurons_per_cluster
                cluster_end = (cluster_idx + 1) * self.neurons_per_cluster
                cluster_state = self.reservoir_state[cluster_start:cluster_end]
                # Compute the prediction for this cluster using its readout.
                pred_value = np.dot(self.Wout_clusters[cluster_idx], cluster_state).item()
                predictions[t, cluster_idx] = pred_value
        return predictions[washout:]

    
