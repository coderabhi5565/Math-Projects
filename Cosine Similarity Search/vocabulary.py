from preprocessing import clean_text


def build_vocabulary(sentences):
    vocabulary = set()

    for sentence in sentences:
        words = clean_text(sentence)

        vocabulary.update(words)

    return sorted(vocabulary)