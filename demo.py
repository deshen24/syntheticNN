import numpy as np
from snn import SyntheticNearestNeighbors 

def normalize_rows(X): 
    return np.array([row/np.linalg.norm(row, 2) for row in X])

def signal_matrix(m, n, r): 
    # user latent features
    U = np.random.randn(m, r) 
    U = normalize_rows(U)

    # item latent features
    V = np.random.randn(n, r)
    V = normalize_rows(V)

    # ratings 
    A = U @ V.T
    return (A, U, V)

# model parameters
m = 100
n = 100
r = 10

# generate model
(A, U, V) = signal_matrix(m, n, r)

# observation (obfuscate last entry)
X = A.copy() 
X[-1, -1] = np.nan 

# estimate via SNN 
params = {
	'n_neighbors': 1,
	'weights': 'distance',
	'verbose': False 
}
snn = SyntheticNearestNeighbors(**params)
X_snn = snn.fit_transform(X)

# report error
delta = np.abs(X_snn[-1,-1] - A[-1,-1])
print("SNN error = {:.2f}".format(delta))










