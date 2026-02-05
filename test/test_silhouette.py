# write your silhouette score unit tests here
import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
# set random seed for reproducibility
np.random.seed(42)
# import kmeans, silhouette, utils
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_basic_silhouette():
    # check if score function works for true mat/label pairing
    num_clusters = 3
    num_feats = 2
    num_samples = 100
    mat, true_labels = make_clusters(n=num_samples, m=num_feats, k=num_clusters, scale=0.5)
    
    silhouette = Silhouette()
    scores = silhouette.score(mat, true_labels)
    assert len(scores) == len(true_labels), 'Cant have more scores than number of samples!'

    # do same for kmeans-returned labels
    
    # create kmeans object now
    km = KMeans(k=num_clusters)
    km.fit(mat)
    km_labels = km.predict(mat)
    km_silhouette = Silhouette()
    km_scores = km_silhouette.score(mat, km_labels)
    assert len(km_scores) == len(km_labels), 'Cant have more scores than number of samples!'

    # check if scores in correct range
    assert km_scores.min() >= -1 and km_scores.max() <= 1, "Scores must be in range(-1,1)"
    assert scores.min() >= -1 and scores.max() <= 1, "Scores must be in range(-1,1)"

    # check if code returns similar answer to sklearn silhouette score
    sklearn_score_true = silhouette_score(mat, true_labels)
    sklearn_score_kmeans = silhouette_score(mat, km_labels)
    tol = 1e-1
    assert np.abs(scores.mean() - sklearn_score_true) <= tol, "Scores dissimilar between sklearn and implementation"
    assert np.abs(km_scores.mean() - sklearn_score_kmeans) <= tol, "Scores dissimilar between sklearn and implementation"

    # check if input mat is correct dimension
    with pytest.raises(ValueError):
        silhouette_error = Silhouette()
        silhouette_error.score(np.array(list(range(5))), y=np.array(list(range(5))))
    
    # check if y length is same as X number of rows
    with pytest.raises(ValueError):
        silhouette_error = Silhouette()
        silhouette_error.score(np.zeros((5,5)), np.zeros(8))
    