import numpy as np

from tfidf import calculate_idf
from vectorizer import tfidf_vector


def build_index(sentences, vocabulary):
    idf = calculate_idf(
        sentences,
        vocabulary
    )

    vectors = []

    for sentence in sentences:
        vector = tfidf_vector(
            sentence,
            vocabulary,
            idf
        )

        vectors.append(vector)

    vectors = np.array(vectors)

    return vectors, idf