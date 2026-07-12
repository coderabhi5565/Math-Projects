import numpy as np


def generate_dataset(num_samples):
    x = np.linspace(0, 10, num_samples)
    y = 2 * x + 1

    return x, y