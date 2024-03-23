import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn import datasets

from clustering_algorithms.c_means import CMeans
from clustering_algorithms.k_means import KMeans

if __name__ == "__main__":
    colors = ["red", "black", "blue"]
    clusters_nb = len(colors)
    points, clusters = datasets.make_blobs(n_samples=1000, n_features=10, centers=clusters_nb)
    iteration_nb = 100
    plt.ion()
    plt.figure()

    def draw_clusters(position: int, title: str, clusters: NDArray) -> None:
        for cluster in range(clusters_nb):
            plt.subplot(1, 3, position)
            plt.scatter(
                points[clusters == cluster][:, 0],
                points[clusters == cluster][:, 1],
                color=colors[cluster],
            )
            plt.pause(0.01)
            plt.title(title)
            plt.draw()

    draw_clusters(1, "original data", clusters)
    k_means = KMeans(clusters_nb, iteration_nb)
    setattr(k_means, "log_iteration", lambda i, c: draw_clusters(2, f"k-means #{i}", c))
    k_means.fit(points)
    c_means = CMeans(clusters_nb, iteration_nb, 2)
    setattr(c_means, "log_iteration", lambda i, c: draw_clusters(3, f"c-means #{i}", c))
    c_means.fit(points)
