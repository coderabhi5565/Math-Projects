from preprocessing import clean_text


def sentence_to_vector(sentence, vocabulary):
    words = clean_text(sentence)

    vector = []

    for vocab_word in vocabulary:
        count = words.count(vocab_word)
        vector.append(count)

    return vector