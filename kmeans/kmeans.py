# import dependencies
import numpy as np

def compute_distance(x_arr, miu_arr, return_clusters=False):
    """
    Calculate the distance between every records of x_arr to the miu_arr
    Args:
    x_arr | numpy.ndarray(m, n):
        data for the distance to be measured from the reference points (miu_arr)
    miu_arr | numpy.ndarray(k, n):
        reference points for measuring distance (centroids)

    Returns:
    distances | numpy.ndarray(m, k):
        the calculated distance between x_arr and miu_arr
    """
    m = x_arr.shape[0]
    k = miu_arr.shape[0]
    distances = np.zeros((m, k))
    for i in range(m):
        for j in range(k):
            diff = x_arr[i]-miu_arr[j]
            distances[i, j] = np.linalg.norm(diff)
    if return_clusters:
        clusters = np.zeros((m,))
        for i in range(m):
            clusters[i] = np.argmin(distances[i, :])
        return clusters.astype('int')
    else:
        return distances


def clusters_mean(x_arr, indices, fetch_k=None):
    """
    Calculate the mean of x_arr at every centroids
    Args:
    x_arr | numpy.ndarray(m, n):
        data to be used for calculating its mean
    indices | numpy.ndarray(m):
        indices or clusters of data x_arr
    fetch_k | int:
        get the mean of x_arr only for specific centroid k

    Returns:
    cmeans | numpy.ndarray(k, n) or (1, n):
        new mean of x_arr based on the centroids
    """
    u_clusters = np.unique(indices)
    m, n = x_arr.shape
    k = u_clusters.shape[0]    
    if fetch_k==None:
        c_means = np.zeros((k, n))
        for idx in range(k):
            c_idx = indices==u_clusters[idx]
            x_idx = x_arr[c_idx, :]
            c_means[idx, :] = x_idx.mean(axis=0)
    else:
        c_idx = indices==fetch_k
        x_idx = x_arr[c_idx, :]
        c_means = np.array([x_idx.mean(axis=0)])
    return c_means

def distortion_cost(x_arr, centroids):
    """
    Calculate the distortion cost on x_arr with the given centroids
    Args:
    x_arr | numpy.ndarray(m, n):
        data to used for calculating the distortion cost
    centroids | numpy.ndarray(k, n):
        centroids of the kmeans algorithm

    Returns:
    cost | float:
        the distortion cost on x_arr with the given centroids
    """
    clusters = compute_distance(x_arr, centroids, return_clusters=True)
    m = x_arr.shape[0]
    cost = 0
    for idx in range(m):
        temp = np.linalg.norm(x_arr[idx, :]-centroids[clusters[idx], :])
        cost += temp
    cost = cost/m
    return cost


def run_kmeans(x_arr, in_centroids, num_iters, printout=False):
    """
    Run kmeans clustering on data matrix x_arr
    Args:
    x_arr | numpy.ndarray(m, n):
        data to used for running kmeans clustering algorithm
    in_centroids | numpy.ndarray(k, n):
        initial centroids to run kmeans
    num_iters | int:
        number of iteration to run kmeans clustering algorithm

    Returns:
    f_centroids | numpy.ndarray(k, n):
        final centroids after running kmeans for num_iters
    clusters_res | numpy.ndarray(m):
        clustering result from kmeans algorithm
    """
    intervals = np.ceil(num_iters/5)
    f_centroids = in_centroids.copy()
    for idx in range(num_iters):
        if (idx%intervals)==0 and printout:
            print(f'KMeans iteration: {idx}')
            print("Cost: {}\n".format(distortion_cost(x_arr, f_centroids)))
        indices = compute_distance(x_arr, f_centroids, return_clusters=True)
        f_centroids = clusters_mean(x_arr, indices)
    if printout:
        print(f"{idx+1} Iterations completed!")
        print("Cost: {}".format(distortion_cost(x_arr, f_centroids)))
    clusters_res = compute_distance(x_arr, f_centroids, return_clusters=True)
    return f_centroids, clusters_res
        
def random_init_centroids(x_arr, k, num_iters=100):
    """
    Generate randomized centroids with the lowest cost function
    Args:
    x_arr | numpy.ndarray(m, n):
        data matrix to used for generating random centroids
    k | int:
        number of centroids to generate
    num_iters | int:
        number of iterations to perform for searching optimum centroids

    Returns:
    f_centroids | numpy.ndarray(k, n):
        optimum centroids with lowest possible cost function
    cost | float:
        distortion cost with respect to f_centroids and x_arr
    
    """
    m, n = x_arr.shape
    cost = np.zeros((num_iters,))
    randomizer = lambda x: np.random.randint(x)
    f_centroids = np.zeros((k, n))
    cost = 0
    for c in range(num_iters):
        init_centroids = np.zeros((k, n))
        for idx in range(k):
            for col in range(n):
                init_centroids[idx, col] = x_arr[randomizer(m),col]
        if c==0:
            cost += distortion_cost(x_arr, init_centroids)
        else:
            temp_cost = distortion_cost(x_arr, init_centroids)
            if temp_cost<cost:
                cost = temp_cost
                f_centroids = init_centroids
    return cost, f_centroids
