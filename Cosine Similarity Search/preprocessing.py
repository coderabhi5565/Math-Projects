import string


def clean_text(text):
    text = text.lower()

    translator = str.maketrans("", "", string.punctuation)

    text = text.translate(translator)

    words = text.split()

    return words