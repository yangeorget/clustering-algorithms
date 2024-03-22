import numpy as np
from numpy.typing import NDArray

from clustering_algorithms.clustering_algorithm import ClusteringAlgorithm


class CMeans(ClusteringAlgorithm):
    """
    A Numpy implementation of the C-Means clustering_algorithms algorithm.
    """

    def __init__(self, k: int, iterations: int, m: int):
        """
        :param k: the number of clusters
        :param iterations: the number of iterations
        :param m: the fuzzifier factor (should be > 1)
        """
        super().__init__(k, iterations)
        self.m = m

    def compute_centroids(self, points: NDArray, weights: NDArray) -> NDArray:
        """
        Computes the centroids given the weights.
        :param points: a nxd matrix
        :param weights: a nxk matrix
        :return: a kxd matrix where k is the number of clusters and d the number of dimensions
        """
        wm = weights**self.m
        return (points.T @ wm / np.sum(wm, axis=0)).T

    def fit(self, points: NDArray) -> NDArray:
        n = points.shape[0]
        weights = np.array(np.random.dirichlet(np.ones(self.k), n))
        for iteration in range(self.iterations):
            centroids = self.compute_centroids(points, weights)
            weights = self.predict(points, centroids)
            clusters = np.argmax(weights, axis=1)
            self.log_iteration(iteration, clusters)
        return centroids

    def predict(self, points: NDArray, centroids: NDArray) -> NDArray:
        points_centroids = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2) ** (-2 / (self.m - 1))
        return points_centroids / np.sum(points_centroids, axis=1, keepdims=True)
