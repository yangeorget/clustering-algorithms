import time

from sklearn import datasets

from clustering_algorithms.c_means import CMeans
from clustering_algorithms.k_means import KMeans

if __name__ == "__main__":
    clusters_nb = 3
    points, clusters = datasets.make_blobs(n_samples=100000, n_features=10, centers=clusters_nb)
    iteration_nb = 100
    start_time = time.time()
    k_means = KMeans(clusters_nb, iteration_nb)
    k_means.fit(points)
    print(f"K-Means took {time.time() - start_time} seconds")
    start_time = time.time()
    c_means = CMeans(clusters_nb, iteration_nb, 2)
    c_means.fit(points)
    print(f"C-Means took {time.time() - start_time} seconds")
