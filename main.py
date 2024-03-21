import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn import datasets

from c_means import CMeans
from k_means import KMeans

if __name__ == "__main__":
    points, clusters = datasets.make_blobs(n_samples=100000)
    colors = ["red", "black", "blue"]  # the scikit learn data come in three clusters
    plt.ion()
    plt.figure()

    def draw_clusters(position: int, title: str, clusters: NDArray) -> None:
        for color_idx in range(len(colors)):
            plt.subplot(1, 3, position)
            plt.scatter(
                points[clusters == color_idx][:, 0],
                points[clusters == color_idx][:, 1],
                color=colors[color_idx],
            )
            plt.pause(0.01)
            plt.title(title)
            plt.draw()

    draw_clusters(1, "original data", clusters)
    k_means = KMeans(len(colors), 100)
    setattr(k_means, "log_iteration", lambda i, c: draw_clusters(2, f"k-means #{i}", c))
    k_means.fit(points)
    c_means = CMeans(len(colors), 100, 2)
    setattr(c_means, "log_iteration", lambda i, c: draw_clusters(3, f"c-means #{i}", c))
    c_means.fit(points)
