# Synthetic Nearest Neighbors (SNN)

Synthetic Nearest Neighbors (SNN) is an algorithm for matrix completion. 

This repository implements the SNN algorithm presented in [Causal Matrix Completion](https://arxiv.org/abs/2109.15154); we remark that the synthetic interventions estimator of [Synthetic Interventions](https://arxiv.org/abs/2006.07691) is a special case of the SNN algorithm. Please refer to the paper for details on the SNN algorithm, as well as its operating assumptions and statistical guarantees. 

Additionally, please contact the authors below if you find any bugs or have any suggestions for improvement (e.g., efficiently storing anchor rows and columns so as to avoid repeated computations). Thank you!

Author: Dennis Shen (deshen@mit.edu, dshen24@berkeley.edu) 

## Code dependencies
This code has the following dependencies:

- Python 3.5+ with the libraries: (networkx, numpy, sklearn)

## Usage

```python
from snn import SyntheticNearestNeighbors

# X is the observed data matrix with missing entries denoted as NaN

# initialize model 
snn = SyntheticNearestNeighbors() 

# impute missing entries
X_snn = snn.fit_transform(X)
```

## Example
We provide an example script in `demo.py`.

