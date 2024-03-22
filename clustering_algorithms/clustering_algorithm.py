from numpy.typing import NDArray


# mypy: disable-error-code=empty-body
class ClusteringAlgorithm:
    """
    An abstract clustering_algorithms algorithm.
    Clustering algorithms produce models that are set of centroids.
    """

    def __init__(self, k: int, iterations: int):
        """
        :param k: the number of clusters
        :param iterations: the number of iterations
        """
        self.iterations = iterations
        self.k = k

    def fit(self, points: NDArray) -> NDArray:
        """
        Fits a model to the training set (the points).
        :param points: the training set
        :return: a model as a set of centroids
        """
        pass

    def predict(self, points: NDArray, model: NDArray) -> NDArray:
        """
        Applies the model to a set of points.
        :param points: a new set of points
        :param model: the clustering_algorithms model
        :return: for each point, the clustering_algorithms output
        """
        pass

    def log_iteration(self, iteration: int, clusters: NDArray) -> None:
        """
        This is where you want to do something when an iteration has completed.
        :param iteration: the iteration number
        :param clusters: the computed clusters
        """
        pass
