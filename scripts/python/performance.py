import time

import matplotlib.pyplot as plt
from sklearn import cluster, datasets

from clustering_algorithms.c_means import CMeans
from clustering_algorithms.k_means import KMeans

if __name__ == "__main__":
    clusters_nb = 3
    iteration_nb = 100
    sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    algorithms = [
        KMeans(k=clusters_nb, iterations=iteration_nb, tolerance=0),
        cluster.KMeans(n_clusters=clusters_nb, max_iter=iteration_nb, tol=0),
        CMeans(k=clusters_nb, iterations=iteration_nb),
    ]
    colors = ["red", "blue", "green"]
    for algorithm, color in zip(algorithms, colors):
        times = []
        for size in sizes:
            points, _ = datasets.make_blobs(n_samples=2**size, n_features=10, centers=clusters_nb)
            start_time = time.time()
            algorithm.fit(points)
            times.append(time.time() - start_time)
        plt.plot(sizes, times, color=color)
    plt.show()
