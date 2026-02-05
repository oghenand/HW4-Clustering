# Write your k-means unit tests here
import pytest
import numpy as np
from scipy.spatial.distance import cdist
# set random seed for reproducibility
np.random.seed(42)
# import kmeans, silhouette, utils
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_kmeans_basic():
    # create synthetic data - 0.5 scale to increase separation
    num_clusters = 3
    num_feats = 2
    num_samples = 100
    mat, true_labels = make_clusters(num_samples=num_samples, m=num_feats, k=num_clusters, scale=0.5)
    # create kmeans object now
    km = KMeans(k=num_clusters)
    km.fit(mat)

    # check if centroids found, labels assigned, errors calculated
    for attr in ['centroids', 'labels', 'error']:
        assert hasattr(km, attr), f"Model should have {attr} after fitting"
    
    # check if centroids the correct shape
    km_centroids = km.get_centroids()
    assert km_centroids.shape == (num_clusters, num_feats), "Centroids must be an array of kxm (num clusters x num feats)"

    # check if labels are correct size
    km_predictions = km.predict(mat)
    assert len(km_predictions) == num_samples == len(true_labels), "Every sample must have a label (i.e. num samples == length of labels)"
    
    # check if correct number of labels found
    assert len(np.unique(km_predictions)) == num_clusters, "Number of unqiue cluster labels much match number of desired clusters"

    # check if labels are ints (should be since labels found by checking idx of min distance)
    assert np.issubdtype(km_predictions.dtype, np.integer)

    # check if label identities within reasonable range (range is (0, num_samples))
    assert km_predictions.min() in range(0,num_samples), "Labels predicted not within bounds"
    assert km_predictions.max() in range(0, num_samples), "Labels predicted not within bounds"

    # check for non-negative error
    assert km.get_error() >= 0, "Error must be non-negative"

    # check that correct cluster is calculated for a random point
    # not exhaustive, but should be true for any random point
    random_point = np.random.choice(num_samples) # pick random point
    random_point_label = km_predictions[random_point] # get label
    min_dist = float('inf') # compare distance from random point to each cluster and then compare
    min_label = ''
    for centroid in range(len(km_centroids)):
        dist_to_random_pt = np.linalg.norm(km_centroids[centroid,:] - mat[random_point,:])
        if dist_to_random_pt < min_dist:
            min_dist = dist_to_random_pt
            min_label = centroid
    assert min_label == random_point_label, "Label mismatch! Label should be the centroid with the smallest distance from the random point"
    
def test_kmeans_edge_case():
    # create synthetic data - 0.5 scale to increase separation
    num_clusters = 3
    num_feats = 5
    num_samples = 20
    mat, true_labels = make_clusters(num_samples=num_samples, m=num_feats, k=num_clusters, scale=0.5)

    # test case to make sure code raises proper ValueError if inelgible k is provided
    for k_val in [num_samples+1, 0, -1]:
        with pytest.raises(ValueError):
            km = KMeans(k=k_val) # cannot take negative, 0, or >num_samples k
            km.fit(mat)
    
    # check if improper shape for mat provided
    km = KMeans(k=3)
    with pytest.raises(ValueError):
        km.fit(np.array(list(range(5))))
    
    # check predict edge cases
    # edge case 1: non-2D np array
    km = KMeans(k=num_clusters)
    km.fit(mat)
    with pytest.raises(ValueError):
        km.predict(np.array(list(range(5))))
    # edge case 2: matrix with diff number of feats
    with pytest.raises(ValueError):
        km.predict(np.zeros((5, num_feats+1)))

    # check for improper order-of-calls
    km_out_of_order = KMeans(k=3)
    # calling predict, get_error and get_centroids before fit
    with pytest.raises(ValueError):
        km_out_of_order.predict(mat)
    with pytest.raises(ValueError):
        km_out_of_order.get_error()
    with pytest.raises(ValueError):
        km_out_of_order.get_centroids()
    