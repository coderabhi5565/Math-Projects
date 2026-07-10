from vectorizer import sentence_to_vector
from cosine import cosine_similarity


def semantic_search(query, sentences, vocabulary):
    query_vector = sentence_to_vector(query, vocabulary)

    best_sentence = ""
    best_score = -1

    for sentence in sentences:
        sentence_vector = sentence_to_vector(sentence, vocabulary)

        score = cosine_similarity(query_vector, sentence_vector)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence, best_score