import numpy as np

from clustering_algorithms.clustering_algorithm import ClusteringAlgorithm


class TestClusteringAlgorithm:
    def test_distances(self) -> None:
        distances = ClusteringAlgorithm(1).distances(np.array([[1, 2], [1, 1]]), np.array([[1, 2], [1, 1]]))
        assert np.all(distances == np.array([[0, 1], [1, 0]]))
