from dataset import sentences
from vocabulary import build_vocabulary
from indexer import build_index
from search import semantic_search

from pca import reduce_dimension
from visualize import visualize_vectors

from vectorizer import tfidf_vector


def main():

    vocabulary = build_vocabulary(
        sentences
    )

    vectors, idf = build_index(
        sentences,
        vocabulary
    )

    while True:

        query = input(
            "\nEnter Query (exit to quit): "
        )

        if query.lower() == "exit":
            break

        results = semantic_search(
            query,
            sentences,
            vectors,
            vocabulary,
            idf
        )

        print()

        print("=" * 60)
        print("Top Results")
        print("=" * 60)

        for sentence, score in results:

            print(
                f"{score:.4f} -> {sentence}"
            )

        reduced_vectors = reduce_dimension(
            vectors
        )

        query_vector = tfidf_vector(
            query,
            vocabulary,
            idf
        )

        query_vector = reduce_dimension(
            query_vector.reshape(1, -1)
        )

        visualize_vectors(
            reduced_vectors,
            sentences,
            query_vector[0]
        )


if __name__ == "__main__":
    main()