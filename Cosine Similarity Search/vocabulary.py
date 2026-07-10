from dataset import sentences

from preprocessing import clean_text


def build_vocabulary(sentences):
    vocabulary = set()

    for sentence in sentences:
        words = clean_text(sentence)

        for word in words:
            vocabulary.add(word)

    return sorted(vocabulary)