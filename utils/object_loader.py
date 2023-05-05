__all__ = [
    'get_object_data'
]

import pywavefront
import numpy as np

def get_object_data(filename: str, dots_limit:int=None):

    scene = pywavefront.Wavefront(
        filename,
        create_materials=True,
        strict=False
    )

    vertices_array = np.array(scene.vertices)
    print('Vertices number loaded:', vertices_array.shape[0])

    if dots_limit:
        indices = np.random.randint(vertices_array.shape[0], size=dots_limit)
        vertices_array = vertices_array[indices, :]
        print('Limited vertices number:', vertices_array.shape[0])
    return vertices_array
