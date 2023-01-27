import numpy as np

class kmeans_model:
    """
    KMeans clustering algorithm.

    kmeans_clustering fits a KMeans algorithm to minimize the distortion
    cost (distance squared from data to centroids) by changing the values
    of centroids.

    Parameters
    ----------
    x_arr -> numpy.ndarray(m, n)
        data to be searched for its cluster.
        
    k -> int
        number of centroids to used for clustering.
        
    random_init_centroids -> bool, default=True
        if True, centroids will be initialized by random sampling with
        the minimum distortion cost.

    num_searchs -> int, default=100
        if random_init_centroids=True, specify how many iterations to perform
        when randomly search initial centroids.
        
    init_centroids -> numpy.ndarray(k, n), default=None
        if random_init_centroids=False, init_centroids must be assign to an
        array of initial centroids.
    """
    def __init__(
        self,
        x_arr,
        k,
        random_init_centroids=True,
        num_searchs=100,
        init_centroids=None
        ):
        self.x_arr = x_arr
        self.k = k
        self.num_searchs = num_searchs
        self.random_init_centroids = random_init_centroids
        self.init_centroids = init_centroids

    def random_search_centroids(self):
        """
        Generate randomized centroids with the lowest distortion cost.

        Returns
        -------
        self -> object
            estimator of random search initial centroids.
        """
        m, n = self.x_arr.shape
        randomizer = lambda x: np.random.randint(x)
        cost = 0
        for i in range(self.num_searchs):
            centroids = np.zeros((self.k, n))
            for row in range(self.k):
                for col in range(n):
                    centroids[row, col] = self.x_arr[randomizer(m), col]
            if i==0:
                cost += self.score(centroids)
            else:
                t_cost = self.score(centroids)
                if t_cost<cost:
                    cost = t_cost
                    self.init_centroids = centroids
        return self

    def compute_distance(self, centroids=None, return_clusters_only=True):
        """
        Calculate the distance between x_arr and the centroids.

        Parameters
        ----------
        return_clusters_only -> bool
            if True, returns only the estimated clusters calculated via
            the minimum distances between x_arr and the centroids.

        centroids -> numpy.ndarray(k,n), default=None
            centroids for calculating the distances and estimating clusters.
            if None, use the init_centroids from the class.

        Returns
        -------
        distances -> numpy.ndarray(k, n)
            the distances from x_arr to every centroid.
            
        clusters -> numpy.ndarray(m)
            the estimated cluster of the x_arr.
        """
        m, n = self.x_arr.shape
        distances = np.zeros((m, self.k))
        if isinstance(centroids, type(None)):
            try:
                centroids = self.init_centroids.copy()
            except Exception as e:
                self.random_search_centroids()
                centroids = self.init_centroids.copy()
        for i in range(m):
            for j in range(self.k):
                diff = self.x_arr[i] - centroids[j]
                distances[i, j] = np.linalg.norm(diff)
        if return_clusters_only:
            clusters = np.zeros((m,), dtype='int')
            for i in range(m):
                clusters[i] = np.argmin(distances[i, :])
            return clusters
        else:
            return distances
                
    def score(self, centroids=None):
        """
        Calculate the distortion cost with the given x_arr and centroids.

        Parameters
        ----------
        centroids -> numpy.ndarray(k, n), default=None
            centroids as a references for calculating the distortion cost.
            if None, use the init_centroids from the class.

        Returns
        -------
        cost -> float
            the calculated distortion cost with the given x_arr and centroids.
        """
        if isinstance(centroids, type(None)):
            try:
                centroids = self.init_centroids.copy()
            except Exception as e:
                self.random_search_centroids()
                centroids = self.init_centroids.copy()
        clusters = self.compute_distance(centroids, return_clusters_only=True)
        m, n = self.x_arr.shape
        cost = 0
        for idx in range(m):
            t_cost = np.linalg.norm(
                self.x_arr[idx, :]-centroids[clusters[idx],:]
                )
            cost += t_cost
        cost /= m
        return cost

    def fit(self, num_iters=100, printout=False, intervals=5):
        """
        Fit KMeans clustering estimator for num_iters

        Parameters
        ----------
        num_iters -> int, default=100
            number of iterations to perform KMeans clustering algorithm.

        printout -> bool, default=False
            if True, will printout the distortion cost at each iteration.

        intervals -> int, default=5
            if num_iters is large (>10), instead of printing distorting cost at
            each iteration, it'll print distortion cost with intervals. if None,
            it'll print distortion cost at each iteration

        Returns
        -------
        self -> object
            fitted KMeans estimator.
        """
        def calc_meansk(x_data, indices, k):
            m, n = x_data.shape
            c_means = np.zeros((k, n))
            for idx in range(k):
                data_idx = x_data[indices==idx, :]
                c_means[idx, :] = np.mean(data_idx, axis=0)
            return c_means
        
        if isinstance(intervals, type(None)):
            intervals = num_iters
        num_int = int(np.ceil(num_iters/intervals))

        # check for initial centroids
        try:
            f_centroids = self.init_centroids.copy()
        except Exception as e:
            self.random_search_centroids()
            f_centroids = self.init_centroids.copy()
        
        for idx in range(num_iters):
            if (idx%num_int)==0 and printout:
                print("KMeans Iteration: {}".format(idx))
                print("Distortion Cost: {}\n".format(self.score(centroids=f_centroids)))
            indices = self.compute_distance(centroids=f_centroids)
            f_centroids = calc_meansk(self.x_arr, indices, self.k)
        if printout:
            print("KMeans Iteration: {}".format(num_iters))
            print("Distortion Cost: {}\n".format(self.score(centroids=f_centroids)))

        self.f_centroids = f_centroids
        return self
