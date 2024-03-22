import numpy as np
from clustering_algorithm import ClusteringAlgorithm
from numpy.typing import NDArray


class CMeans(ClusteringAlgorithm):
    """
    A Numpy implementation of the C-Means clustering algorithm.
    """

    def __init__(self, k: int, iterations: int, m: int):
        super().__init__(k, iterations)
        self.m = m  # fuzzifier

    def compute_centroids(self, points: NDArray, weights: NDArray) -> NDArray:
        """
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
