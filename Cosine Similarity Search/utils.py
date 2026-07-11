import numpy as np


def print_line(length=60):
    print("=" * length)


def print_title(title):
    print_line()
    print(title)
    print_line()


def normalize(vector):
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm