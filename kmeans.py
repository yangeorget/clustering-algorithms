import numpy as np


class Kmeans:
    def __init__(self, k, iterations, draw_clusters=None):
        self.iterations = iterations
        self.k = k
        self.draw_clusters = draw_clusters

    def initialize_centroids(self, points):
        return points[np.random.choice(points.shape[0], self.k, replace=False), :]

    def create_clusters(self, centroids, points):
        n = points.shape[0]
        return np.array(
            [
                np.argmin([np.linalg.norm(centroids[j] - points[i]) for j in range(self.k)])
                for i in range(n)
            ]
        )

    def compute_centroids(self, cluster_indices, points):
        return np.array(
            [
                np.mean(points[cluster_indices == j], axis=0)
                for j in range(self.k)
            ]
        )

    def compute_clusters(self, points):
        centroids = self.initialize_centroids(points)
        for iteration in range(self.iterations):
            clusters = self.create_clusters(centroids, points)
            new_centroids = self.compute_centroids(clusters, points)
            # if np.all(new_centroids == centroids):
            #    return clusters
            centroids = new_centroids
            if self.draw_clusters:
                self.draw_clusters(f"kmeans #{iteration}", clusters)
        return clusters, centroids
