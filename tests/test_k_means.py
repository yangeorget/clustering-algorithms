import numpy as np
import pytest

from clustering_algorithms.clustering_exception import ClusteringException
from clustering_algorithms.k_means import KMeans


class TestKMeans:

    def test_compute_centroids_1_1(self) -> None:
        centroids = KMeans(1).compute_centroids(np.array([[1, 2]]), np.array([0]))
        assert np.all(centroids == np.array([[1, 2]]))

    def test_compute_centroids_1_2(self) -> None:
        centroids = KMeans(1).compute_centroids(np.array([[1, 3], [5, 3]]), np.array([0, 0]))
        assert np.all(centroids == np.array([[3, 3]]))

    def test_compute_centroids_2_2(self) -> None:
        centroids = KMeans(2).compute_centroids(np.array([[1, 3], [5, 3]]), np.array([0, 1]))
        assert np.all(centroids == np.array([[1, 3], [5, 3]]))

    def test_fit_1(self) -> None:
        centroids, clusters = KMeans(1, 2).fit(np.array([[1, 3], [5, 3]]))
        assert np.all(centroids == np.array([[3, 3]]))
        assert np.all(clusters == np.array([0, 0]))

    def test_fit_2(self) -> None:
        centroids, clusters = KMeans(2, 1).fit(np.array([[1, 3], [5, 3]]))
        assert np.all(np.sort(centroids, axis=0) == np.array([[1, 3], [5, 3]]))
        assert np.all(np.sort(clusters, axis=0) == np.array([0, 1]))

    def test_fit_2_1_too_many_clusters_exception(self) -> None:
        with pytest.raises(ClusteringException) as exception:
            KMeans(2).fit(np.array([[1, 2]]))
        assert str(exception.value) == "Too many clusters requested"

    def test__fit_too_few_clusters_exception(self) -> None:
        with pytest.raises(ClusteringException) as exception:
            KMeans(3, 4)._fit(
                np.array([[0.5, -2], [0, 0], [0.5, 0], [1, 0], [1.1, 0], [3.1, 0], [3.2, 0]]),
                np.array([[0.5, -2], [3.1, 0], [3.2, 0]]),
            )
        assert str(exception.value) == "Too few clusters computed"

    def test_predict(self) -> None:
        clusters = KMeans(1).predict(np.array([[1, 3], [5, 3]]), np.array([3, 3]))
        assert np.all(clusters == np.array([[0, 0]]))
