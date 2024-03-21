import numpy as np
from numpy.typing import NDArray

from clustering_model import ClusteringAlgorithm


class CMeans(ClusteringAlgorithm):
    """
    A Numpy implementation of the C-Means clustering algorithm.
    The clustering model consists in a nxk weight matrix where n is the number of points and k the number of clusters.
    """
    def __init__(self, k: int, iterations: int, m: int):
        super().__init__(k, iterations)
        self.m = m  # fuzzifier

    def initialize_model(self, points: NDArray) -> NDArray:
        n = points.shape[0]
        return np.array(np.random.dirichlet(np.ones(self.k), n))

    def compute_centroids(self, points: NDArray, weights: NDArray) -> NDArray:
        """
        :param points: a nxd matrix
        :param weights: a nxk matrix
        :return: a kxd matrix where k is the number of clusters and d the number of dimensions
        """
        wm = weights**self.m
        return (points.T @ wm / np.sum(wm, axis=0)).T

    def predict(self, points: NDArray, centroids: NDArray) -> NDArray:
        """
        :param points: a nxd matrix
        :param centroids: a kxd matrix
        :return: a nxk weight matrix
        """
        points_centroids = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2) ** (-2 / (self.m - 1))
        return points_centroids / np.sum(points_centroids, axis=1, keepdims=True)

    def update_model(self, points: NDArray, weights: NDArray, iteration: int) -> NDArray:
        centroids = self.compute_centroids(points, weights)
        weights = self.predict(points, centroids)
        self.log_iteration(iteration, np.argmax(weights, axis=1))
        return weights
