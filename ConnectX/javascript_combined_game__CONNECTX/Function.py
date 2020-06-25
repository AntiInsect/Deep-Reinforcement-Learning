import numpy as np


def coordinates_set(width, height):
    """
    Get a set of coordinate tuples with width and height.
    :param width: width.
    :param height: height.
    :return: <set (x_axis, y_axis)>
    """
    s = set()
    for i in range(width):
        for j in range(height):
            s.add((i, j))
    return s


def get_data_augmentation(array: np.ndarray, operation=lambda a: a):
    """
    Get data augmentation.
    :param array: the data to augment.
    :param operation: What to do after data augmentation.
    :return: Augmented data.
    """

    # A single number.
    if array.shape == ():
        return np.zeros(8) + array

    return [operation(array),
            operation(np.rot90(array, 1)),
            operation(np.rot90(array, 2)),
            operation(np.rot90(array, 3)),
            operation(np.fliplr(array)),
            operation(np.rot90(np.fliplr(array), 1)),
            operation(np.rot90(np.fliplr(array), 2)),
            operation(np.rot90(np.fliplr(array), 3))]
