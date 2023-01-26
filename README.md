# kmeans-clustering-built-with-python
Unsupervised learning is one of the branch on machine learning, nowadays the most commonly used algorithm for unsupervised learning (specifically in clustering problems) is KMeans Clustering Algorithm. The key concept of the clustering learning is trying to minimize the distortion cost (or the distances of the x data to the centroids). And in doing so, we need to make initial guess of the centroids (preferably one with a low distortion cost). From there, the algorithm will then look for x data and assign each of data points to a cluster (measured by the lowest distance between that data point to a specific centroid compared to other centroids). After that, we then: 1) Compute the mean of every cluster; 2) Move all the centroids to its cluster's mean. We want to iterate this process until the distortion cost remains the same (meaning: no change at all).
