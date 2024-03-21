import numpy as np
from numpy.typing import NDArray

from clustering_model import ClusteringAlgorithm


class KMeans(ClusteringAlgorithm):
    """
    A Numpy implementation of the K-Means clustering algorithm.
    The clustering model consists in a nxk weight matrix where n is the number of points and k the number of clusters.
    """
    def initialize_model(self, points: NDArray) -> NDArray:
        n = points.shape[0]
        return points[np.random.choice(n, self.k, replace=False), :]

    def compute_centroids(self, points: NDArray, clusters: NDArray) -> NDArray:
        return np.array([np.mean(points[clusters == j], axis=0) for j in range(self.k)])

    def predict(self, points: NDArray, centroids: NDArray) -> NDArray:
        n = points.shape[0]
        return np.array([np.argmin(np.linalg.norm(centroids - points[i], axis=1)) for i in range(n)])

    def update_model(self, points: NDArray, centroids: NDArray, iteration: int) -> NDArray:
        clusters = self.predict(points, centroids)
        centroids = self.compute_centroids(points, clusters)
        self.log_iteration(iteration, clusters)
        return centroids

    def should_stop(self, centroids: NDArray, new_centroids: NDArray) -> bool:
        return (new_centroids == centroids).all()
