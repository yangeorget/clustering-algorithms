import numpy as np
from numpy.typing import NDArray

from clustering_algorithms.clustering_algorithm import ClusteringAlgorithm


class KMeans(ClusteringAlgorithm):
    """
    A Numpy implementation of the K-Means clustering_algorithms algorithm.
    """

    def compute_centroids(self, points: NDArray, clusters: NDArray) -> NDArray:
        """
        Computes the centroid given the clusters.
        :param points: a nxd matrix
        :param clusters: a nx1 matrix
        :return: a kxd matrix where k is the number of clusters and d the number of dimensions
        """
        return np.array([np.mean(points[clusters == j], axis=0) for j in range(self.k)])

    def fit(self, points: NDArray) -> NDArray:
        n = points.shape[0]
        centroids = points[np.random.choice(n, self.k, replace=False), :]
        for iteration in range(self.iterations):
            clusters = self.predict(points, centroids)
            self.log_iteration(iteration, clusters)
            new_centroids = self.compute_centroids(points, clusters)
            if self.should_stop(centroids, new_centroids):
                break
            centroids = new_centroids
        return centroids

    def predict(self, points: NDArray, centroids: NDArray) -> NDArray:
        n = points.shape[0]
        return np.array([np.argmin(np.linalg.norm(centroids - points[i], axis=1)) for i in range(n)])

    def should_stop(self, centroids: NDArray, new_centroids: NDArray) -> bool:
        return np.all(new_centroids == centroids)  # type: ignore
