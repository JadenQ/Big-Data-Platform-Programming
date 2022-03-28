from datetime import datetime
import sys
import numpy as np
from sklearn.cluster import KMeans

EPS = np.finfo(float).eps

class bmm:
	def __init__(self, n_components, iters = 50, threshold = 1e-2,verbose = False):
		# number of clusters
		self.n_components = n_components
		# max iterations
		self.iters = iters
		# convergence threshold
		self.threshold = threshold
		# print report or not
		self.verbose = verbose
		# cluster number
		k = self.n_components
		# pi
		self.weights = np.array([1 / k for _ in range(k)])
		# q
		self.means = None
		# converge condition
		self.converged_ = False

	def fit(self,x, means = None, labels = None):
		k = self.n_components
		n = x.shape[0]
		d = x.shape[1]

		## initialization ##

		# self.means = np.ndarray(shape = (k,d))
		# use kmeans to initialize mean
		self.means = kmeans_init(x, k, means = means, verbose = self.verbose)
		
		start = datetime.now()
		iterations = 0
		# convergence setup
		prev_log_likelihood = None
		cur_log_likelihood = -np.inf

		# iterate
		while iterations <= self.iters:
			elapsed = datetime.now() - start

			## E step ##
			# cal the new log_likelihood and responsibility: γ
			log_likelihood, gamma = self.score_compute(x)
			cur_log_likelihood = log_likelihood.mean()

			# report
			if self.verbose:
				print('[{:02d}] likelihood = {} (elapsed {})'
                      .format(iterations, cur_log_likelihood, elapsed))
			# converge or not
			if prev_log_likelihood is not None:
				change = abs(cur_log_likelihood - prev_log_likelihood)
				if change < self.threshold:
					self.converged_ = True
					break

			## M step ##
			self.m_step(x, gamma)
			iterations += 1

        # convergence
		end = datetime.now()
		elapsed = end - start
		print('converged in {} iterations in {}'.format(iterations, elapsed))

	def predict(self, x):
		return np.sum(np.exp(self.log_P(x)), 1)



	# calculate the component of γ: p(x_n | q_k) in a log version
	def log_P(self, x):

		k = self.n_components
		pi = self.weights 
		q = self.means
		x_c = 1 - x
		q_c = 1 - q
		log_support = np.ndarray(shape=(x.shape[0], k))

		for i in range(k):
			log_support[:, i] = (
				np.sum(x * np.log(q[i, :].clip(min=1e-50)), 1) \
                + np.sum(x_c * np.log(q_c[i, :].clip(min=1e-50)), 1))

		return log_support

    # calculate the γ： responsibility
	def score_compute(self, x):

		log_support = self.log_P(x)

		lpr = log_support + np.log(self.weights)
		logprob = np.logaddexp.reduce(lpr, axis=1)
		responsibilities = np.exp(lpr - logprob[:, np.newaxis])

		return logprob, responsibilities

	# using matrix to calculate p_i and q_k
	def m_step(self, x, z):
		# z is gamma - responsibility
		weights = z.sum(axis=0)
		weighted_x_sum = np.dot(z.T, x)
		inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

		self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
		self.means = weighted_x_sum * inverse_weights

# initialize using kmeans
def kmeans_init(x, k, means = None, verbose = False):
	if means is None:
		kmeans = KMeans(n_clusters = k, 
			verbose = int(verbose)).fit(x).cluster_centers_
	else:
		kmeans = means[:k,:]
	return kmeans
