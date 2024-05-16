from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray


# mypy: disable-error-code=empty-body
class ClusteringAlgorithm:
    """
    An abstract clustering algorithm.
    Clustering algorithms produce models that are set of centroids.
    """

    def __init__(self, k: int, iterations: int = 1):
        """
        :param k: the number of clusters
        :param iterations: the number of iterations
        """
        self.iterations = iterations
        self.k = k

    def init(self, points: NDArray, method: str = "random") -> Union[NDArray, Sequence[NDArray]]:
        """
        Inits the model.
        :param points: the training set
        :param method: the init method, defaults to "random"
        :return: a sequence of arrays starting with, at least: a set of centroids
        :raise ClusteringException when the cluster computation fails
        """
        pass

    def fit(self, points: NDArray) -> Sequence[NDArray]:
        """
        Fits a model to the training set (the points).
        :param points: the training set
        :return: a sequence of arrays starting with, at least : a set of centroids, the corresponding clusters
        :raise ClusteringException when the cluster computation fails
        """
        pass

    def predict(self, points: NDArray, centroids: NDArray) -> NDArray:
        """
        Applies the model to a set of points.
        :param points: a new set of points
        :param centroids: the centroids
        :return: for each point, the clustering output
        """
        pass

    def distances(self, points: NDArray, centroids: NDArray) -> NDArray:
        """
        Computes the distances between points and centroids.
        :param points: a nxd matrix
        :param centroids: a kxd matrix
        :return: the distances as a nxk matrix
        """
        return np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)

    def log_iteration(self, iteration: int, centroids: NDArray, clusters: NDArray) -> None:
        """
        This is where you want to do something when an iteration has completed.
        :param iteration: the iteration number
        :param centroids: the computed centroids
        :param clusters: the computed clusters
        """
        pass
