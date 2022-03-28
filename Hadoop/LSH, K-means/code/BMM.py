# '''
# Bernoulli likelihood and BMM algorithm in MINIST binary dataset. 
# '''
# import numpy as np
# import mixture

# random_seed = 2021


# # PARAMETERS
# K = 10
# D = 28*28
# N = 10000 # training set:60000

# # 

import numpy as np

import mixture



class bmm(mixture.mixture):

    def __init__(self, n_components, covariance_type='diag',
                 n_iter=100, verbose=False):

        super().__init__(n_components, covariance_type=covariance_type,
                         n_iter=n_iter, verbose=verbose)

    def _log_support(self, x):

        k = self.n_components; pi = self.weights; mu = self.means

        x_c = 1 - x
        mu_c = 1 - mu

        log_support = np.ndarray(shape=(x.shape[0], k))

        for i in range(k):
            log_support[:, i] = (
                np.sum(x * np.log(mu[i, :].clip(min=1e-50)), 1) \
                + np.sum(x_c * np.log(mu_c[i, :].clip(min=1e-50)), 1))

        return log_support
    
    
    

class BMMclassifier:

    def __init__(self, n_components,
                 means_init_heuristic='random',
                 covariance_type='diag', means=None, verbose=False):

        self.n_components = n_components
        self.means_init_heuristic = means_init_heuristic
        self.covariance_type = covariance_type
        self.means = means
        self.verbose = verbose

        self.models = dict()

    def fit(self, x, labels):

        label_set = set(labels)

        for label in label_set:

            x_subset = x[np.in1d(labels, label)]

            self.models[label] = bmm(
                self.n_components, covariance_type=self.covariance_type,
                verbose=self.verbose)

            print('training label {} ({} samples)'
                  .format(label, x_subset.shape[0]))

            self.models[label].fit(
                x_subset, means_init_heuristic=self.means_init_heuristic,
                means=self.means)

    def predict(self, x, label_set):

        highest_likelihood = 0
        likeliest_label = None

        n = x.shape[0]

        likelihoods = np.ndarray(shape=(len(label_set), n))

        for label in label_set:
            print('predicting label:', label)
            likelihoods[label] = self.models[label].predict(x)

        predicted_labels = np.argmax(likelihoods, axis=0)

        return predicted_labels
