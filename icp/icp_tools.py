from typing import List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
import os, psutil
from numba import cuda, float64, int32
import math

def best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def nearest_neighbor(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    distances = cdist(source, target)
    indices = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(source.shape[0]), indices]
    # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    
    return min_distances, indices

def nearest_neighbor_split(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    batches = 10
    batch_size = source.shape[0] // batches
    indices: np.ndarray = None
    min_distances: np.ndarray = None
    for i in range(batches+1):
        distances = cdist(source[i*batch_size:(i+1)*batch_size], target)
        indices_batch = np.argmin(distances, axis=1)
        min_distances_batch = distances[np.arange(source[i*batch_size:(i+1)*batch_size].shape[0]), indices_batch]
        if i == 0:
            indices = indices_batch
            min_distances = min_distances_batch
        else:
            # indices_batch = indices_batch+batch_size*i
            indices = np.hstack((indices, indices_batch))
            min_distances = np.hstack((min_distances, min_distances_batch))

        # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    
    return min_distances, indices

@cuda.jit(device=True)
def find_nearest_neighbor_cuda(point, dst, min_distance, min_index):
    min_distance[0] = math.inf
    min_index[0] = -1
    for j in range(dst.shape[0]):
        distance = 0
        for d in range(3):
            diff = point[d] - dst[j, d]
            distance += diff * diff
        if distance < min_distance[0]:
            min_distance[0] = distance
            min_index[0] = j

@cuda.jit
def find_nearest_neighbors_kernel_cuda(src, dst, min_distances, indices):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= src.shape[0]:
        return

    min_distance = cuda.local.array(1, float64)
    min_index = cuda.local.array(1, int32)

    find_nearest_neighbor_cuda(src[i], dst, min_distance, min_index)

    min_distances[i] = min_distance[0]
    indices[i] = min_index[0]

def nearest_neighbor_cuda(src, dst, threads_per_block: int = None):
    # src = src.astype(np.float32)
    # dst = dst.astype(np.float32)

    min_distances = np.empty(src.shape[0], dtype=np.float64)
    indices = np.empty(src.shape[0], dtype=np.int32)

    threadsperblock = 256 if not threads_per_block else threads_per_block
    blockspergrid = (src.shape[0] + (threadsperblock - 1)) // threadsperblock
    find_nearest_neighbors_kernel_cuda[blockspergrid, threadsperblock](src, dst, min_distances, indices)

    return min_distances, indices
