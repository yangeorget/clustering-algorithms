import numpy as np


class Cmeans:
    def __init__(self, k, iterations, m, draw_clusters=None):
        self.iterations = iterations
        self.k = k  # number of clusters
        self.m = m  # fuzzifier
        self.draw_clusters = draw_clusters

    def initialize_weights(self, points):
        n = points.shape[0]
        return np.array(np.random.dirichlet(np.ones(self.k), n))

    def compute_centroids(self, weights, points):
        """
        :return: a (k,d) shaped array
        """
        n = points.shape[0]
        d = points.shape[1]
        centroids = np.zeros((self.k, d))
        for j in range(self.k):
            denominator = np.sum(np.power(weights[:, j], self.m))
            centroids[j] = np.sum(np.multiply(points, np.power(weights[:, j], self.m).reshape(n,1)), axis=0) / denominator
        return centroids

    def compute_weights(self, points, centroids):
        """
        :return: a (n,k) shaped array
        """
        n = points.shape[0]
        weights = np.zeros((n, self.k))
        for i in range(n):
            denominator = np.sum(np.power(np.linalg.norm(centroids - points[i], axis=1), -2 / (self.m - 1)), axis=0)
            for j in range(self.k):
                weights[i, j] = np.power(np.linalg.norm(centroids[j] - points[i]), -2 / (self.m - 1)) / denominator
        return weights

    def compute_clusters(self, points):
        weights = self.initialize_weights(points)
        for iteration in range(self.iterations):
            centroids = self.compute_centroids(weights, points)
            weights = self.compute_weights(points, centroids)
            clusters = np.argmax(weights, axis=1)
            if self.draw_clusters:
                self.draw_clusters(f"cmeans #{iteration}", clusters)
        return clusters, centroids, weights
