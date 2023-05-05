import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prepare_plot_data(data, box_limit:int=1):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    # Draw data
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=1)
    ax.set_xlim(-box_limit, box_limit)
    ax.set_ylim(-box_limit, box_limit)
    ax.set_zlim(-box_limit, box_limit)
    ax.set_box_aspect([1, 1, 1])

    # Plot axes with arrows indicating their positive directions
    origin = [0, 0, 0]
    x_axis = [box_limit, 0, 0]
    y_axis = [0, box_limit, 0]
    z_axis = [0, 0, box_limit]
    ax.quiver(*origin, *x_axis, color='r')
    ax.quiver(*origin, *y_axis, color='g')
    ax.quiver(*origin, *z_axis, color='b')
    return ax