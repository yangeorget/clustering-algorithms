from cmeans import Cmeans
from kmeans import Kmeans
from sklearn import datasets
import matplotlib.pyplot as plt

if __name__ == "__main__":
    colors = ["red", "black", "blue"]
    X, y = datasets.make_blobs(n_samples=100000)
    plt.ion()
    plt.figure()

    def draw_clusters(position, title, clusters):
        for color_idx in range(len(colors)):
            plt.subplot(1, 3, position)
            plt.scatter(
                X[clusters == color_idx][:, 0],
                X[clusters == color_idx][:, 1],
                color=colors[color_idx],
            )
            plt.pause(0.01)
            plt.title(title)
            plt.draw()

    draw_clusters(1, "original data", y)
    Kmeans(len(colors), 30, lambda t, c: draw_clusters(2, t, c)).compute_clusters(X)
    Cmeans(len(colors), 30, 2, lambda t, c: draw_clusters(3, t, c)).compute_clusters(X)
