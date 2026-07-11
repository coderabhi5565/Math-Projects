import matplotlib.pyplot as plt
import numpy as np


def visualize_vectors(vectors, sentences, query_vector=None):

    x = vectors[:, 0]
    y = vectors[:, 1]

    plt.figure(figsize=(10, 8))

    plt.scatter(x, y)

    for i, sentence in enumerate(sentences):
        plt.text(
            x[i],
            y[i],
            str(i + 1),
            fontsize=9
        )

    if query_vector is not None:

        plt.scatter(
            query_vector[0],
            query_vector[1],
            marker="X",
            s=200,
            label="Query"
        )

        plt.legend()

    plt.title("Sentence Vector Space")

    plt.xlabel("Dimension 1")

    plt.ylabel("Dimension 2")

    plt.grid(True)

    plt.show()