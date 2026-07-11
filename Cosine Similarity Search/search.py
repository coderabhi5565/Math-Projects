import numpy as np

from cosine import cosine_similarity
from vectorizer import tfidf_vector


def semantic_search(
        query,
        sentences,
        vectors,
        vocabulary,
        idf,
        top_k=3
):

    query_vector = tfidf_vector(
        query,
        vocabulary,
        idf
    )

    scores = []

    for sentence, vector in zip(
            sentences,
            vectors
    ):

        score = cosine_similarity(
            query_vector,
            vector
        )

        scores.append(
            (sentence, score)
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:top_k]