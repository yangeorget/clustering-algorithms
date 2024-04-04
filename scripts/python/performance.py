import time

import matplotlib.pyplot as plt
from sklearn import datasets

from clustering_algorithms.c_means import CMeans
from clustering_algorithms.k_means import KMeans

if __name__ == "__main__":
    clusters_nb = 3
    iteration_nb = 100
    sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    k_means_times = []
    c_means_times = []
    for size in sizes:
        points, clusters = datasets.make_blobs(n_samples=2**size, n_features=10, centers=clusters_nb)
        start_time = time.time()
        k_means = KMeans(clusters_nb, iteration_nb)
        k_means.fit(points)
        k_means_time = time.time() - start_time
        print(f"K-Means took {k_means_time} seconds")
        k_means_times.append(k_means_time)
        start_time = time.time()
        c_means = CMeans(clusters_nb, iteration_nb)
        c_means.fit(points)
        c_means_time = time.time() - start_time
        print(f"C-Means took {c_means_time} seconds")
        c_means_times.append(c_means_time)
    plt.plot(sizes, k_means_times, color="red")
    plt.plot(sizes, c_means_times, color="blue")
    plt.show()
