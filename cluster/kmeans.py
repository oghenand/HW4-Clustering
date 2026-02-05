import numpy as np
from scipy.spatial.distance import cdist
# set random seed for reproducibility
np.random.seed(42)

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # check if vars are correct type (add more cases for tol and max_iter)
        if not isinstance(k, int):
            raise TypeError('k must be an integer!')
        if not isinstance(tol, float):
            raise TypeError('tol must be a float!')
        if not isinstance(max_iter, int):
            raise TypeError('max_iter must be an integer!')

        # store k, tol and max_iter
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # check if mat is ndim = 2
        if np.ndim(mat) != 2:
            raise ValueError('Features matrix must be 2D')

        # store the data into the class and its attributes (i.e. number of samples and features)
        self.n_samples, self.n_features = mat.shape

        # check if k <= n_samples
        if self.k > self.n_samples or self.k < 1:
            raise ValueError('In this implementation, k must be >=1 and <= number of samples in dataset')
        
        # initialize centroids randomly - returns k centroids
        random_idx = np.random.choice(self.n_samples, self.k, replace=False)
        # get centroids - extract rows of the randomly chosen indices
        self.centroids = mat[random_idx]

        # iterate untile max iterations
        for _ in range(self.max_iter):
            # first calculate distances between each centroid and sample
            distances = cdist(self.centroids, mat, metric='euclidean')
            # assign labels
            labels = np.argmin(distances, axis=0)

            # find new centroids:
            # first initialize
            new_centroids = np.zeros((self.k, self.n_features))
            for k_i in range(self.k):
                # find sample idxs in cluster
                cluster_idxs = mat[labels == k_i]
                # check if cluster is empty and if not, find new cluster center
                if len(cluster_idxs) > 0:
                    new_centroids[k_i] = cluster_idxs.mean(axis=0)
                else:
                    new_centroids[k_i] = self.centroids[k_i]
            
            # find max change between new and old centroids
            max_change = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))

            self.centroids = new_centroids
            self.labels = labels
            # max change less or equal to tol, break
            if max_change <= self.tol:
                break
        
        # calculate errors
        # extract final distances and compute squared errors
        final_distances = cdist(self.centroids, mat, metric='euclidean')
        dist_to_centroids = np.min(final_distances, axis=0) # find distance to the centroid for each sample
        # get mean squared error
        self.error = np.mean(dist_to_centroids**2)

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # check if centroids even exist -- only do once fit and should be doine before predict
        if not hasattr(self, 'centroids'):
            raise ValueError('KMeans has not been fit yet!')

        # check if mat is not 2d
        if np.ndim(mat) != 2:
            raise ValueError('Features matrix must be 2D')
        
        # check if mat has incorrect number of features -- diff number of samples is okay!
        if mat.shape[1] != self.n_features:
            raise ValueError('Number of features must be same as what KMeans was fit on')

        # calculate distances and return labels given centroids from self.fit()
        distances = cdist(self.centroids, mat, metric='euclidean')
        return np.argmin(distances, axis=0)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        # check if erros even exist -- only do once fit
        if not hasattr(self, 'errors'):
            raise ValueError('KMeans has not been fit yet!')
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # check if centroids even exist -- only do once fit
        if not hasattr(self, 'centroids'):
            raise ValueError('KMeans has not been fit yet!')
        return self.centroids

