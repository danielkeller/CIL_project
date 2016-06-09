#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np
import scipy.sparse as sp

from numpy.math cimport INFINITY
cdef inline double square(double x): return x * x

cdef get_assignments(data, double[:,::1] centroids):
    cdef:
        int M = data.shape[0]
        int N = data.shape[1]
        int K = centroids.shape[0]
        np.ndarray[np.int32_t, ndim=1] assignments = np.zeros(M, dtype=np.int32)
        np.ndarray[np.double_t, ndim=1] dists = np.zeros(K)
        int[::1] data_ptr = data.indptr
        int[::1] data_ind = data.indices
        double[::1] data_data = data.data
        #Loop variables MUST BE TYPED!
        int user, ptr, cluster
        double diff, dist, closest_d
    for user in range(M):
        closest_d = INFINITY
        for cluster in range(K):
            dist = 0
            for ptr in range(data_ptr[user], data_ptr[user+1]):
                dist += square(centroids[cluster, data_ind[ptr]] - data_data[ptr])
            if dist < closest_d:
                closest_d = dist
                assignments[user] = cluster
    return assignments

cdef get_centroids(int K, data, int[::1] assignments):
    cdef:
        int M = data.shape[0]
        int N = data.shape[1]
        np.ndarray[np.double_t, ndim=2] centroids = np.zeros((K, N))
        np.ndarray[np.double_t, ndim=2] counts = np.zeros((K, N))
        int[::1] data_ptr = data.indptr
        int[::1] data_ind = data.indices
        double[::1] data_data = data.data
        int user, ptr
    for user in range(M):
        for ptr in range(data_ptr[user], data_ptr[user+1]):
            centroids[assignments[user], data_ind[ptr]] += data_data[ptr]
            counts[assignments[user], data_ind[ptr]] += 1
    counts[counts == 0] = 1
    return centroids / counts

cdef rmse(test, double[:,::1] centroids, int[::1] assignments):
    cdef:
        int M = test.shape[0]
        double err = 0.0
        int user, ptr
        int[::1] test_ptr = test.indptr
        int[::1] test_ind = test.indices
        double[::1] test_data = test.data
    for user in range(M):
        for ptr in range(test_ptr[user], test_ptr[user+1]):
            err += square(test_data[ptr] - centroids[assignments[user], test_ind[ptr]])
    return np.sqrt(err / test.nnz)

def k_means(data, test, K=20, tol=0.01):
    if not sp.isspmatrix_csr(data):
        raise TypeError("data must be a csr matrix")
    if not sp.isspmatrix_csr(test):
        raise TypeError("test must be a csr matrix")
    if data.shape != test.shape:
        raise TypeError("data and test must have the same shape")

    M, N = data.shape
    err = 999
    
    centroids = np.random.rand(K, N) * 6 + 0.5
    while True:
        assignments = get_assignments(data, centroids)
        centroids = get_centroids(K, data, assignments)
        newerr = rmse(test, centroids, assignments)
        if (err - newerr) / err < tol:
            break
        err = newerr
    return err, centroids, assignments
