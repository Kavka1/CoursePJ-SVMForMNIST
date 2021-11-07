from typing import Dict, List, Union, Tuple
import numpy as np


def Histogram_Intersection_Kernel(X: np.array, Y: np.array) -> np.array:
    """Histogram Intrection Kernel is widely used in application of image classification
    Equation:
        k(x, y) = Sum_k(min(x_k, y_k))

    Args:
        X (np.array): samples with shape [num_samples, 28*28]
        Y (np.array): samples with shape [num_samples, 28*28]

    Returns:
        Kernel matrix: K(X, Y) = [[k(x_1, y_1), k(x_1, y_2), k(x_1, y_3)...], ..., [k(x_n, y_1), k(x_n, y_2), ...]]
    """
    kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))          # Initialize kernel matrix
    for i, x in enumerate(X):                                   # for every sample in X, like x_1 with shape (28*28, )
        kernel_matrix[i] = np.sum(np.minimum(x, Y), axis = 1)   # compare x_i to y_1 ~ y_n and get 1 row of kernel 
    return kernel_matrix                                        # return the kernel matrix with shape [sample_num, sample_num]