# %% [1. Introduction to Classifiers]
# Learn Naive Bayes and Logistic Regression for text classification.

# Setup: pip install nltk numpy matplotlib scikit-learn
# NLTK Data: python -m nltk.downloader punkt stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords

def run_classifiers_demo():
    # %% [2. Synthetic Retail Text Data]
    train_reviews = [
        ("This laptop is great! I love the fast processor.", "positive"),
        ("The screen is vibrant and solid.", "positive"),
        ("The battery life is terrible.", "negative"),
        ("Poor performance, bad purchase.", "negative")
    ]
    test_reviews = [
        "This laptop from TechCorp is great! I love the fast processor.",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp."
    ]
    train_texts, train_labels = zip(*train_reviews)
    print("Synthetic Text: Retail product reviews created")
    print(f"Training Data: {len(train_reviews)} labeled reviews")
    print(f"Test Reviews: {test_reviews}")

    # %% [3. TF-IDF Vectorization]
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_reviews)
    print(f"TF-IDF: Training matrix shape {X_train.shape}, Test matrix shape {X_test.shape}")

    # %% [4. Naive Bayes Classifier]
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, train_labels)
    nb_predictions = nb_classifier.predict(X_test)
    print("Naive Bayes: Predictions made")
    for i, pred in enumerate(nb_predictions):
        print(f"Review {i+1}: {pred}")

    # %% [5. Logistic Regression Classifier]
    lr_classifier = LogisticRegression()
    lr_classifier.fit(X_train, train_labels)
    lr_predictions = lr_classifier.predict(X_test)
    print("Logistic Regression: Predictions made")
    for i, pred in enumerate(lr_predictions):
        print(f"Review {i+1}: {pred}")

    # %% [6. Visualization]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    nb_counts = Counter(nb_predictions)
    plt.bar(nb_counts.keys(), nb_counts.values(), color='blue')
    plt.title("Naive Bayes Predictions")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.subplot(1, 2, 2)
    lr_counts = Counter(lr_predictions)
    plt.bar(lr_counts.keys(), lr_counts.values(), color='green')
    plt.title("Logistic Regression Predictions")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("classifiers_output.png")
    print("Visualization: Classifier predictions saved as classifiers_output.png")

    # %% [7. Interview Scenario: Classifiers]
    """
    Interview Scenario: Classifiers
    Q: Compare Naive Bayes and Logistic Regression for text classification.
    A: Naive Bayes assumes feature independence, is faster; Logistic Regression models feature interactions, often more accurate.
    Key: Both effective for text; choice depends on data and complexity.
    Example: MultinomialNB().fit(X_train, y_train)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_classifiers_demo()