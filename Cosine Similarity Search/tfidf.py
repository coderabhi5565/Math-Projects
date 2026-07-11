import math
import numpy as np

from preprocessing import clean_text


def calculate_tf(words, vocabulary):
    tf = np.zeros(len(vocabulary))

    total_words = len(words)

    for i, vocab_word in enumerate(vocabulary):
        tf[i] = words.count(vocab_word) / total_words

    return tf


def calculate_document_frequency(sentences, vocabulary):
    df = np.zeros(len(vocabulary))

    for sentence in sentences:
        words = set(clean_text(sentence))

        for i, vocab_word in enumerate(vocabulary):
            if vocab_word in words:
                df[i] += 1

    return df


def calculate_idf(sentences, vocabulary):
    N = len(sentences)

    df = calculate_document_frequency(sentences, vocabulary)

    idf = np.log((N + 1) / (df + 1)) + 1

    return idf


def calculate_tfidf(sentence, vocabulary, idf):
    words = clean_text(sentence)

    tf = calculate_tf(words, vocabulary)

    return tf * idf