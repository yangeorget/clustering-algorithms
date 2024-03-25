import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn import datasets

from clustering_algorithms.c_means import CMeans
from clustering_algorithms.k_means import KMeans

if __name__ == "__main__":
    colors = ["red", "black", "blue"]
    clusters_nb = len(colors)
    points, clusters = datasets.make_blobs(n_samples=1000, n_features=2, centers=clusters_nb)
    iteration_nb = 100
    plt.ion()
    plt.figure()

    def draw_clusters(position: int, title: str, clstrs: NDArray) -> None:
        for clstr in range(clusters_nb):
            plt.subplot(1, 3, position)
            plt.scatter(
                points[clstrs == clstr][:, 0],
                points[clstrs == clstr][:, 1],
                color=colors[clstr],
            )
            plt.pause(0.01)
            plt.title(title)
            plt.draw()

    draw_clusters(1, "original data", clusters)
    k_means = KMeans(clusters_nb, iteration_nb)
    setattr(k_means, "log_iteration", lambda it, _, c: draw_clusters(2, f"k-means #{it}", c))
    k_means.fit(points)
    c_means = CMeans(clusters_nb, iteration_nb)
    setattr(c_means, "log_iteration", lambda it, _, c: draw_clusters(3, f"c-means #{it}", c))
    c_means.fit(points)
