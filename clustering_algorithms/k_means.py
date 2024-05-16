from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from clustering_algorithms.clustering_algorithm import ClusteringAlgorithm
from clustering_algorithms.clustering_exception import (
    TOO_FEW_CLUSTERS,
    TOO_MANY_CLUSTERS,
    ClusteringException,
)


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

    def init(self, points: NDArray, method: str = "kmeans++") -> NDArray:
        n = points.shape[0]
        if n < self.k:
            raise ClusteringException(TOO_MANY_CLUSTERS)
        if method == "random":
            return points[np.random.choice(n, self.k, replace=False), :]
        else:  # kmeans++
            centroids = np.zeros((self.k, points.shape[1]))
            centroids[0] = points[np.random.choice(n)]
            min_distances = None
            for i in range(1, self.k):
                centroid_distances = self.distances(points, centroids[i - 1]).flatten()
                if min_distances is not None:
                    min_distances = np.min(np.vstack((min_distances, centroid_distances)), axis=0)
                else:
                    min_distances = centroid_distances
                squared_min_distances = min_distances**2
                centroids[i] = points[np.random.choice(n, p=squared_min_distances / np.sum(squared_min_distances))]
            return centroids

    def fit(self, points: NDArray) -> Sequence[NDArray]:
        centroids = self.init(points)
        return self._fit(points, centroids)

    def _fit(self, points: NDArray, centroids: NDArray) -> Sequence[NDArray]:
        for iteration in range(self.iterations):
            clusters = self.predict(points, centroids)
            if len(np.unique(clusters)) < self.k:
                raise ClusteringException(TOO_FEW_CLUSTERS)
            self.log_iteration(iteration, centroids, clusters)
            new_centroids = self.compute_centroids(points, clusters)
            if np.all(new_centroids == centroids):  # type: ignore
                break
            centroids = new_centroids
        return centroids, clusters

    def predict(self, points: NDArray, centroids: NDArray) -> NDArray:
        return np.argmin(self.distances(points, centroids), axis=1)
