from numpy.typing import NDArray


# mypy: disable-error-code=empty-body
class ClusteringAlgorithm:
    """
    An abstract clustering algorithm.
    """
    def __init__(self, k: int, iterations: int):
        self.iterations = iterations
        self.k = k

    def initialize_model(self, points: NDArray) -> NDArray:
        pass

    def update_model(self, points: NDArray, model: NDArray, iteration: int) -> NDArray:
        pass

    def should_stop(self, model: NDArray, new_model: NDArray) -> bool:
        """
        :param model: the previous model
        :param new_model: a new model
        :return: a boolean indicating if the new model is good enough to stop the computation
        """
        return False

    def fit(self, points: NDArray) -> NDArray:
        """
        Fits a model to the training set (the points).
        :param points: the training set
        :return: a model
        """
        model = self.initialize_model(points)
        for iteration in range(self.iterations):
            new_model = self.update_model(points, model, iteration)
            if self.should_stop(model, new_model):
                break
            model = new_model
        return model

    def predict(self, points: NDArray, model: NDArray) -> NDArray:
        """
        Applies the model to a set of points.
        :param points: a new set of points
        :param model: the clustering model
        :return: for each point, the clustering output
        """
        pass

    def log_iteration(self, iteration: int, clusters: NDArray) -> None:
        """
        This is where you want to do something when an iteration has completed.
        :param iteration: the iteration number
        :param clusters: the computed clusters
        """
        pass
