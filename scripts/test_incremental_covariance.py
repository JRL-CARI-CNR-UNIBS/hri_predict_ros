from data_analysis.utils import IncrementalCovariance
import numpy as np
import time

n_features = 2
n_samples = 100

# Create a random matrix of shape (n_samples, n_features)
X = np.random.rand(n_samples, n_features)
print("X:\n", X)

# Compute the covariance matrix all in once using numpy
tic = time.time()
covariance = np.cov(X, rowvar=False)
toc_numpy = (time.time() - tic) * 1e6


# Create an instance of the IncrementalCovariance class
tic = time.time()
ic = IncrementalCovariance(n_features)

for i in range(n_samples):
    # Update the covariance matrix incrementally
    ic.update(X[i])

# Get the incremental covariance matrix
incremental_covariance = ic.cov
toc_incremental = (time.time() - tic) * 1e6

# Compare the two covariance matrices
print("\nCov(X) using numpy [time: ", toc_numpy, " us]:\n", covariance)
print("\nCov(X) using IncrementalCovariance [time: ", toc_incremental, " us]:\n", incremental_covariance)

# Check if the two matrices are equal
assert np.allclose(covariance, incremental_covariance), "Covariance matrices are not equal"