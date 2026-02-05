import numpy as np
from scipy.spatial.distance import cdist
# set random seed for reproducibility
np.random.seed(42)


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if np.ndim(X) != 2:
            raise ValueError('X must be 2D')
        if len(y) != X.shape[0]:
            raise ValueError('Number of observations must be equal between labels array and matrix')

        # extract number of samples and features as useful vars
        n_samples = X.shape[0]
        # initiate scores array
        s_scores = np.zeros(n_samples)
        # calculate pairwise distances (to loop up distances quickly and not have to recalculate)
        distances = cdist(X, X, metric='euclidean')

        # loop through each sample
        for i in range(n_samples):
            sample_cluster_label = y[i]
            # find cluster size
            cluster_size = np.sum(y==sample_cluster_label)
            # check edge case: cluster size = 1 returns silhouette score of zero
            if cluster_size == 1:
                s_scores[i] = 0
                continue
            
            # first find a
            # sum distances to members of same cluster and then divide to find a
            a = np.sum(distances[i, y == sample_cluster_label]) / (cluster_size - 1)

            # now find b
            unique_clusters = np.unique(y[y!=sample_cluster_label]) # find all other clusters
            # iterate through each cluster and find avg distance, then find min of this

            # s_score of zero if only 1 cluster in dataset:
            if len(unique_clusters) == 0:
                s_scores[i] = 0
                continue

            b = float('inf')
            for other_cluster in unique_clusters:
                other_cluster_size = np.sum(y==other_cluster) # get size of other cluster
                avg_dist_to_clust = np.sum(distances[i, y==other_cluster]) / other_cluster_size # compute avg dist
                if avg_dist_to_clust < b: # assign b
                    b = avg_dist_to_clust
            # assign silhoutte score
            s_scores[i] = (b-a)/(max(a,b))
        
        return s_scores

