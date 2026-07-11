import numpy as np

from preprocessing import clean_text
from tfidf import calculate_tfidf


def count_vector(sentence, vocabulary):
    words = clean_text(sentence)

    vector = np.zeros(len(vocabulary))

    for i, vocab_word in enumerate(vocabulary):
        vector[i] = words.count(vocab_word)

    return vector


def tfidf_vector(sentence, vocabulary, idf):
    return calculate_tfidf(
        sentence,
        vocabulary,
        idf
    )