from dataset import sentences
from vocabulary import build_vocabulary
from search import semantic_search

vocabulary = build_vocabulary(sentences)

query = input("Enter your query : ")

result, score = semantic_search(query, sentences, vocabulary)

print("\nMost Similar Sentence")
print(result)
print(f"Similarity Score : {score:.4f}")