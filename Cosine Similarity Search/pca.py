import numpy as np


def reduce_dimension(matrix):

    matrix = matrix - np.mean(
        matrix,
        axis=0
    )

    covariance_matrix = np.cov(
        matrix,
        rowvar=False
    )

    eigen_values, eigen_vectors = np.linalg.eigh(
        covariance_matrix
    )

    sorted_index = np.argsort(
        eigen_values
    )[::-1]

    eigen_vectors = eigen_vectors[
        :,
        sorted_index
    ]

    principal_components = eigen_vectors[:, :2]

    reduced_matrix = np.dot(
        matrix,
        principal_components
    )

    return reduced_matrix