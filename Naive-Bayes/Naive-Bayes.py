import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
 
# 1. Create a synthetic, clearly separable dataset
positive_words = ["good", "excellent", "fantastic", "amazing", "wonderful", "love", "great", "awesome", "nice", "pleasant"]
negative_words = ["bad", "terrible", "awful", "horrible", "hate", "worst", "boring", "poor", "dull", "disappointing"]
 
np.random.seed(0)
 
# 20 positive and 20 negative samples
texts = []
labels = []
for _ in range(20):
    sample = " ".join(np.random.choice(positive_words, size=3, replace=True))
    texts.append(sample)
    labels.append(1)
for _ in range(20):
    sample = " ".join(np.random.choice(negative_words, size=3, replace=True))
    texts.append(sample)
    labels.append(0)
 
# 2. Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
 
# 3. Train Naive Bayes
nb = MultinomialNB()
nb.fit(X, labels)
 
# 4. Prepare test samples
test_texts = [
    "good excellent fantastic",   # positive
    "bad terrible awful",         # negative
    "amazing wonderful love",     # positive
    "boring dull disappointing",  # negative
    "awesome awesome bad",        # mixed
    "worst pleasant horrible",    # mixed
]
X_test = vectorizer.transform(test_texts)
y_pred = nb.predict(X_test)
 
# 5. Project to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_all = np.vstack([X.toarray(), X_test.toarray()])
X_all_2d = pca.fit_transform(X_all)
X_2d = X_all_2d[:len(X.toarray())]
X_test_2d = X_all_2d[len(X.toarray()):]
 
# 6. Plot training data
plt.figure(figsize=(8, 6))
colors = ['red' if label == 1 else 'blue' for label in labels]
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, label='Training data')
 
# 7. Plot test data
test_colors = ['green' if pred == 1 else 'purple' for pred in y_pred]
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=test_colors, marker='*', s=200, edgecolor='k', label='Test data')
for i, txt in enumerate(test_texts):
    plt.annotate(f"{txt}\n(pred={y_pred[i]})", (X_test_2d[i, 0]+0.2, X_test_2d[i, 1]), fontsize=8, color=test_colors[i])
 
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Naive Bayes Text Classification Visualization (Synthetic Data)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# 8. Output predictions
for text, pred in zip(test_texts, y_pred):
    print(f'"{text}": predicted class {pred}')