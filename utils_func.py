import numpy as np
import matplotlib.pyplot as plt

#TRAINING AND TEST SET SPLIT
def train_test_split(X, y, dates, train_ratio=0.8):
    """
    Split data into training and test sets, including dates/time.
    
    Parameters:
    X (ndarray): Input features
    y (ndarray): Target values
    dates (pd.Series or ndarray): Dates corresponding to each sample
    train_ratio (float): Proportion of data to use for training (default 0.8)
    
    Returns:
    tuple: X_train, y_train, X_test, y_test, train_dates, test_dates
    """
    assert 0 < train_ratio < 1, "Train ratio must be between 0 and 1."
    
    # Get the number of samples
    n_samples = len(X)
    
    # Determine the split index
    train_index = int(train_ratio * n_samples)
    
    # Split the data
    X_train, y_train = X[:train_index], y[:train_index]
    X_test, y_test = X[train_index:], y[train_index:]
    
    # Split dates
    train_dates = dates[:train_index]
    test_dates = dates[train_index:]
    
    return X_train, y_train, X_test, y_test, train_dates, test_dates


# NRMSE
def compute_nrmse(predictions, actuals):
    mse = np.mean((predictions - actuals) ** 2, axis=0)  # Mean Squared Error per output
    rmse = np.sqrt(mse)                                  # Root Mean Squared Error
    nrmse = rmse / (actuals.max(axis=0) - actuals.min(axis=0))  # Normalized by range of actual values for each output
    return nrmse

#To measure state redundancy
def average_state_change(esn, X):
    """
    Compute the average norm of state change between successive time steps
    for EchoStateNetworkWithAdaptiveClusters.
    """
    # Reset reservoir before measuring
    esn.reset_reservoir()

    n_steps = len(X)
    states = np.zeros((n_steps, esn.n_reservoir))
    for t, x in enumerate(X):
        esn._update_reservoir(x)
        states[t] = esn.reservoir_state

    diffs = np.linalg.norm(states[1:] - states[:-1], axis=1)
    return np.mean(diffs), states
    

#State correlation matrix    
def plot_state_correlation_matrix(states, title="Reservoir State Correlation Matrix"):
    """
    Plot correlation matrix of reservoir neuron activations.

    Parameters
    ----------
    states : ndarray of shape (timesteps, n_reservoir)
        Reservoir states over time.
    title : str
        Plot title.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Compute correlation between neurons
    C = np.corrcoef(states.T)

    # Optional: set diagonal to zero for clearer contrast
    #np.fill_diagonal(C, 0)

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(C, cmap='coolwarm', center=0, square=True,
                     cbar_kws={"label": r"$\gamma$"})
    plt.title(title, fontsize=18)
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")

    # Set tick marks every 20 neurons
    n = C.shape[0]
    tick_positions = np.arange(0, n, 20)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions, rotation=0)
    ax.set_yticklabels(tick_positions, rotation=0)

    plt.tight_layout()
    plt.savefig("ring_corr_sun.eps",dpi = 300)
    plt.show()