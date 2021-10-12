"""
Synthetic Nearest Neighbors algorithm
"""
import sys
import warnings 
import random
import numpy as np 
import networkx as nx
from networkx.algorithms.clique import find_cliques
from sklearn.utils import check_array

class SyntheticNearestNeighbors(): 
	"""
	Impute missing entries in a matrix via SNN algorithm
	"""
	def __init__(
			self,
			n_neighbors=1, 
			weights='uniform',
			random_splits=False,
			max_rank=None,
			spectral_t=None,
			linear_span_eps=0.1,
			subspace_eps=0.1, 
			min_value=None,
			max_value=None,
			verbose=True): 
		"""
		Parameters
		----------
		n_neighbors : int 
		Number of synthetic neighbors to construct

		weights : str 
		Weight function used in prediction. Possible values: 
		(a) 'uniform': each synthetic neighbor is weighted equally 
		(b) 'distance': weigh points inversely with distance (as per train error)

		random_splits : bool 
		Randomize donors prior to splitting  

		max_rank : int 
		Perform truncated SVD on training data with this value as its rank 

		spectral_t : float 
		Perform truncated SVD on training data with (100*thresh)% of spectral energy retained. 
		If omitted, then the default value is chosen via Donoho & Gavish '14 paper. 

		linear_span_eps : float
		If the (normalized) train error is greater than (100*linear_span_eps)%,
		then the missing pair fails the linear span test. 

		subspace_eps : float
		If the test vector (used for predictions) does not lie within (100*subspace_eps)% of 
		the span covered by the training vectors (used to build the model), 
		then the missing pair fails the subspace inclusion test. 

		min_value : float 
		Minumum possible imputed value 

		max_value : float 
		Maximum possible imputed value 

		verbose : bool 
		"""
		self.n_neighbors = n_neighbors 
		self.weights = weights
		self.random_splits = random_splits
		self.max_rank = max_rank 
		self.spectral_t = spectral_t
		self.linear_span_eps = linear_span_eps
		self.subspace_eps = subspace_eps
		self.min_value = min_value 
		self.max_value = max_value 
		self.verbose = verbose

	def __repr__(self):
		""" 
		print parameters of SNN class
		"""
		return str(self)

	def __str__(self):
		field_list = []
		for (k, v) in sorted(self.__dict__.items()):
			if (v is None) or (isinstance(v, (float, int))):
				field_list.append("%s=%s" % (k, v))
			elif isinstance(v, str):
				field_list.append("%s='%s'" % (k, v))
		return "%s(%s)" % (
			self.__class__.__name__,", ".join(field_list))

	def _check_input_matrix(self, X, missing_mask):
		"""
		check to make sure that the input matrix 
		and its mask of missing values are valid.
		"""
		if len(X.shape)!=2:
			raise ValueError(
				"expected 2d matrix, got %s array" % (X.shape,)
			)
		(m, n) = X.shape 
		if not len(missing_mask)>0: 
			warnings.simplefilter("always")
			warnings.warn(
				"input matrix is not missing any values"
			)
		if len(missing_mask)==int(m*n): 
			raise ValueError(
				"input matrix must have some observed (i.e., non-missing) values"
			)

	def _prepare_input_data(self, X, missing_mask):
		"""
		prepare input matrix X. return if valid else terminate 
		"""
		X = check_array(X, force_all_finite=False)
		if (X.dtype!="f") and (X.dtype!="d"):
			X = X.astype(float)
		self._check_input_matrix(X, missing_mask)
		return X

	def _check_weights(self, weights):
		"""
		check to make sure weights are valid
		"""
		if weights not in ("uniform", "distance"):
			raise ValueError(
				"weights not recognized: should be 'uniform' or 'distance'"
			)
		return weights

	def _split(self, arr, k):
		"""
		split array arr into k subgroups of roughly equal size
		"""
		(m, n) = divmod(len(arr), k)
		return (arr[i*m + min(i, n): (i+1)*m + min(i+1, n)] for i in range(k))

	def _find_anchors(self, X, missing_pair): 
		"""
		find model learning submatrix by reducing to max biclique problem
		"""
		(missing_row, missing_col) = missing_pair
		obs_rows = np.argwhere(~np.isnan(X[:, missing_col])).flatten()
		obs_cols = np.argwhere(~np.isnan(X[missing_row, :])).flatten()

		# dennis: make sure (i,j) not in (obs_rows, obs_cols)

		# create bipartite incidence matrix 
		B = X[obs_rows]
		B = B[:, obs_cols]
		if not np.any(np.isnan(B)): # check if fully connected already
			return (obs_rows, obs_cols)
		B[np.isnan(B)] = 0 

		# bipartite graph 
		(n_rows, n_cols) = B.shape 
		A = np.block([[np.ones((n_rows, n_rows)), B],
		  			  [B.T, np.ones((n_cols, n_cols))]])
		G = nx.from_numpy_matrix(A)

		# find max clique that yields the most square (nxn) matrix
		cliques = list(find_cliques(G))
		d_min = 0 
		max_clique_rows_idx = False
		max_clique_cols_idx = False
		for clique in cliques: 
			clique = np.sort(clique)
			clique_rows_idx = clique[clique<n_rows]
			clique_cols_idx = clique[clique>=n_rows] - n_rows
			d = min(len(clique_rows_idx), len(clique_cols_idx))
			if d>d_min: 
				d_min = d 
				max_clique_rows_idx = clique_rows_idx
				max_clique_cols_idx = clique_cols_idx

		# determine model learning rows & cols 
		anchor_rows = obs_rows[max_clique_rows_idx]
		anchor_cols = obs_cols[max_clique_cols_idx]
		return (anchor_rows, anchor_cols)

	def _spectral_rank(self, s):
		"""
		retain all singular values that compose at least (100*self.spectral_t)% spectral energy
		"""
		if self.spectral_t==1.0: 
			rank = len(s)
		else: 
			total_energy = (s**2).cumsum() / (s**2).sum()
			rank = list((total_energy>self.spectral_t)).index(True) + 1
		return rank

	def _universal_rank(self, s, ratio): 
		"""
		retain all singular values above optimal threshold as per Donoho & Gavish '14:
		https://arxiv.org/pdf/1305.5870.pdf
		""" 
		omega = 0.56*ratio**3 - 0.95*ratio**2 + 1.43 + 1.82*ratio
		t = omega * np.median(s) 
		rank = max(len(s[s>t]), 1)
		return rank 

	def _pcr(self, X, y):
		"""
		principal component regression (PCR) 
		"""
		(u, s, v) = np.linalg.svd(X, full_matrices=False)
		if self.max_rank is not None: 
			rank = self.max_rank 
		elif self.spectral_t is not None: 
			rank = self._spectral_rank(s)
		else: 
			(m, n) = X.shape 
			rank = self._universal_rank(s, ratio=m/n)
		s_rank = s[:rank]
		u_rank = u[:, :rank]
		v_rank = v[:rank, :] 
		beta = ((v_rank.T/s_rank) @ u_rank.T) @ y
		return (beta, u_rank, s_rank, v_rank)

	def _clip(self, x):
		"""
		clip values to fall within range [min_value, max_value]
		"""
		if self.min_value is not None:
			x = self.min_value if x<self.min_value else x 
		if self.max_value is not None:
			x = self.max_value if x>self.max_value else x 
		return x

	def _train_error(self, X, y, beta): 
		"""
		compute (normalized) training error
		""" 
		y_pred = X @ beta 
		delta = np.linalg.norm(y_pred-y)
		ratio = delta / np.linalg.norm(y)
		return ratio**2

	def _subspace_inclusion(self, V1, X2):
		"""
		compute subspace inclusion statistic 
		"""  
		delta = (np.eye(V1.shape[1]) - (V1.T@V1)) @ X2 
		ratio = np.linalg.norm(delta) / np.linalg.norm(X2)
		return ratio**2

	def _isfeasible(self, train_error, subspace_inclusion_stat): 
		"""
		check feasibility of prediction
		True iff linear span + subspace inclusion tests both pass
		""" 
		# linear span test
		ls_feasible = True if train_error<=self.linear_span_eps else False 

		# subspace test
		s_feasible = True if subspace_inclusion_stat<=self.subspace_eps else False 
		return True if (ls_feasible and s_feasible) else False 

	def _synth_neighbor(self, X, missing_pair, anchor_rows, anchor_cols, covariates=None): 
		"""
		construct the k-th synthetic neighbor 
		"""
		# initialize
		(missing_row, missing_col) = missing_pair
		y1 = X[missing_row, anchor_cols]
		X1 = X[anchor_rows, :] 
		X1 = X1[:, anchor_cols]
		X2 = X[anchor_rows, missing_col]

		# add covariates
		if covariates is not None: 
			y1_covariates = np.hstack([y1, covariates[missing_row]])
			X1_covariates = np.hstack([X1, covariates[anchor_rows]])
		else:
			y1_covariates = y1.copy()
			X1_covariates = X1.copy()

		# learn k-th synthetic neighbor
		(beta, _, s_rank, v_rank) = self._pcr(X1_covariates.T, y1_covariates)

		# prediction
		pred = self._clip(X2@beta) 

		# diagnostics 
		train_error = self._train_error(X1.T, y1, beta)
		subspace_inclusion_stat = self._subspace_inclusion(v_rank, X2) 
		feasible = self._isfeasible(train_error, subspace_inclusion_stat)

		# assign weight of k-th synthetic neighbor
		if self.weights=='uniform':
			weight = 1 
		elif self.weights=='distance':
			d = train_error + subspace_inclusion_stat
			weight = 1/d if d>0 else sys.float_info.max
		return (pred, feasible, weight)

	def _predict(self, X, missing_pair, covariates=None): 
		""" 
		combine predictions from all synthetic neighbors 
		"""
		# find anchor rows and cols
		(anchor_rows, anchor_cols) = self._find_anchors(X, missing_pair=missing_pair) 
		if not anchor_rows.size: 
			(pred, feasible) = (np.nan, False)
		else: 
			if self.random_splits:
				anchor_rows = np.random.permutation(anchor_rows)
			anchor_rows_splits = list(self._split(anchor_rows, 
												  k=self.n_neighbors))
			pred = np.zeros(self.n_neighbors)
			feasible = np.zeros(self.n_neighbors)
			w = np.zeros(self.n_neighbors)

			# iterate through all row splits
			for (k, anchor_rows_k) in enumerate(anchor_rows_splits): 
				(pred[k], feasible[k], w[k]) = self._synth_neighbor(X,
														            missing_pair=missing_pair, 
														            anchor_rows=anchor_rows_k, 
														   		    anchor_cols=anchor_cols,
														   		    covariates=covariates)
			w /= np.sum(w)
			pred = np.average(pred, weights=w)
			feasible = all(feasible)
		return (pred, feasible)

	def fit_transform(self, X, covariates=None, test_set=None): 
		"""
		complete missing entries in matrix 
		""" 
		# get missing entries to impute
		missing_set = test_set if test_set is not None else np.argwhere(np.isnan(X))
		num_missing = len(missing_set)

		# check and prepare data 
		X = self._prepare_input_data(X, missing_set)

		# check weights
		self.weights = self._check_weights(self.weights)

		# initialize 
		X_imputed = X.copy() 
		std_matrix = np.zeros(X.shape)
		self.feasible = np.empty(X.shape)
		self.feasible.fill(np.nan)

		# complete missing entries 
		for (i, missing_pair) in enumerate(missing_set): 
			if self.verbose: 
				print("[SNN] iteration {} of {}".format(i+1, num_missing))

			# predict missing entry
			(pred, feasible) = self._predict(X, 
											 missing_pair=missing_pair,
									 	     covariates=covariates)

			# store in imputed matrices
			(missing_row, missing_col) = missing_pair
			X_imputed[missing_row, missing_col] = pred 
			self.feasible[missing_row, missing_col] = feasible

		if self.verbose:
			print("[SNN] complete")
		return X_imputed




