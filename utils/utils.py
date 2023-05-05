__all__ = [
    'time_record',
    'get_max_coord',
    'beautiful_print_progress',
    'rotate_3d_data'
]

from functools import wraps
import time

import numpy as np

def time_record(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print('Time elapsed for func', func.__name__, ':', time.perf_counter()-start)
        return result
    return wrapper

def get_max_coord(data):
    return max(np.amax(data), (np.amin(data)))

def beautiful_print_progress(iteration: int, mean_error: float, prev_error: float):
    if iteration == 1:
        print(f' iteration_n | {"mean_error":^20} | {"difference":^20} ')
    print(f' {iteration:^11} | {mean_error:^20} | {(mean_error - prev_error):^20}')

def rotate_3d_data(data):
    # fig = plt.figure(figsize=(5, 5), dpi=100)
    # ax = fig.add_subplot(111, projection='3d')

    # Generate random angles for the x, y, and z axes
    theta_x = np.random.uniform(0, 360)
    theta_y = np.random.uniform(0, 360)
    theta_z = np.random.uniform(0, 360)

    print('Rotate x by', theta_x, 'degrees')
    print('Rotate y by', theta_y, 'degrees')
    print('Rotate z by', theta_z, 'degrees')

    # Convert the angles to radians
    theta_x = np.radians(theta_x)
    theta_y = np.radians(theta_y)
    theta_z = np.radians(theta_z)

    # Generate the rotation matrices for each axis
    R_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

    # Apply the rotation matrices to the data
    rotated_data = data @ R_x.T @ R_y.T @ R_z.T

    # directional_vector = [1, 0, 0]
    # rotated_directional_vector = directional_vector @ R_x.T @ R_y.T @ R_z.T

    return rotated_data
