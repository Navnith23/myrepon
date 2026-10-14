import math
from collections import Counter
doc1 = "I love programming in Python"
doc2 = "Python programming is fun"
word1 = doc1.lower().split()
word2 = doc2.lower().split()
vocabulary = set(word1 + word2)
v1 = Counter(word1)
v2 = Counter(word2)
vector1 = [v1[word] for word in vocabulary]
vector2 = [v2[word] for word in vocabulary]
dot_prod = sum(a*b for a,b in zip(vector1, vector2))
mag1 = sum(a*a for a in vector1) ** 0.5
mag2 = sum(b*b for b in vector2) ** 0.5
cosine_sim = dot_prod / (mag1 * mag2) if mag1 * mag2 != 0 else 0
inter = set(word1) & set(word2)
union = set(word1) | set(word2)
jaccard_sim = len(inter) / len(union) if len(union) != 0 else 0
print("Document 1:", doc1)
print("Document 2:", doc2)
print("Vocabulary:", vocabulary)
print("vector1:", vector1)
print("vector2:", vector2)
print("Cosine Similarity:", round(cosine_sim, 2))
print("Jaccard Similarity:", round(jaccard_sim, 2)) 