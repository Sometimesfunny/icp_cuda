
from typing import List, Tuple
import numpy as np

from .icp_tools import nearest_neighbor, best_fit_transform, nearest_neighbor_split
from ..utils import beautiful_print_progress, time_record

class ICP_Solo:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.source = source
        self.target = target
        print(self.source.dtype)
        self.mean_error_history: List[float] = []
        self.iteration: int = 0
        self.transformation_matrix: np.ndarray = None
        self.distances: np.ndarray = []
        self.transformed_matrix: np.ndarray = None

    @time_record
    def icp(self, max_iterations: int = 100, tolerance: float = 1e-6, verbose: bool = True, split_heighbors: bool = False) -> Tuple[np.ndarray, List[float], int]:
        A = self.source
        B = self.target
        
        if A is None:
            raise ValueError('source array is None')
        if B is None:
            raise ValueError('target array is None')

        src = np.ones((4, A.shape[0]))
        src[:3, :] = A.T
        src3d = src[:3, :]
        dst = np.ones((4, B.shape[0]))
        dst[:3, :] = B.T

        prev_error = 0
        mean_error = 0
        self.mean_error_history = []
        self.iteration = 0
        self.distances = None
        self.transformed_matrix = None

        stop_reason = ''

        for i in range(max_iterations):
            if split_heighbors:
                self.distances, indices = nearest_neighbor_split(src3d.T, B)
            else:
                self.distances, indices = nearest_neighbor(src3d.T, B)

            dst_chorder = dst[:3, indices]

            self.transformation_matrix = best_fit_transform(src3d.T, dst_chorder.T)

            src = self.transformation_matrix @ src
            src3d = src[:3, :]

            mean_error = np.mean(self.distances)
            self.mean_error_history.append(mean_error)
            self.iteration = i + 1
            if verbose:
                beautiful_print_progress(self.iteration, mean_error, prev_error)
            if abs(prev_error - mean_error) < tolerance:
                stop_reason = 'Convergence reached'
                break
            prev_error = mean_error
        else:
            stop_reason = 'Max iterations reached'

        self.transformation_matrix = best_fit_transform(A, src3d.T)
        result = (self.transformation_matrix, self.distances, self.iteration)

        print('Stop reason:', stop_reason)
        return result
    
    def transform(self) -> np.ndarray:
        A = self.source
        
        if A is None:
            raise ValueError('source array is None')
        
        if self.transformation_matrix is None:
            ValueError('No transformation matrix found. Perform icp() first')

        return (self.transformation_matrix @ np.vstack((A.T, np.ones((1, A.shape[0])))))[:3, :].T

