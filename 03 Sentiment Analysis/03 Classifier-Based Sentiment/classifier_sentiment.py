# %% [1. Introduction to Classifier-Based Sentiment]
# Learn Naive Bayes and SVM classifiers for sentiment analysis with NLTK and scikit-learn.

# Setup: pip install nltk numpy matplotlib scikit-learn
# NLTK Data: python -m nltk.downloader punkt
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from collections import Counter

def word_feats(words):
    return {word: True for word in words}

def run_classifier_sentiment_demo():
    # %% [2. Synthetic Retail Text Data]
    train_data = [
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
    print("Synthetic Text: Retail product reviews created")
    print(f"Training Data: {len(train_data)} labeled reviews")
    print(f"Test Reviews: {test_reviews}")

    # %% [3. Naive Bayes Classifier]
    train_set = [(word_feats(nltk.word_tokenize(text)), label) for text, label in train_data]
    nb_classifier = NaiveBayesClassifier.train(train_set)
    nb_predictions = [nb_classifier.classify(word_feats(nltk.word_tokenize(review))) for review in test_reviews]
    print("Naive Bayes: Predictions made")
    for i, pred in enumerate(nb_predictions):
        print(f"Review {i+1}: {pred}")

    # %% [4. SVM Classifier]
    svm_classifier = SklearnClassifier(LinearSVC()).train(train_set)
    svm_predictions = [svm_classifier.classify(word_feats(nltk.word_tokenize(review))) for review in test_reviews]
    print("SVM: Predictions made")
    for i, pred in enumerate(svm_predictions):
        print(f"Review {i+1}: {pred}")

    # %% [5. Visualization]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    nb_counts = Counter(nb_predictions)
    plt.bar(nb_counts.keys(), nb_counts.values(), color='blue')
    plt.title("Naive Bayes Predictions")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.subplot(1, 2, 2)
    svm_counts = Counter(svm_predictions)
    plt.bar(svm_counts.keys(), svm_counts.values(), color='green')
    plt.title("SVM Predictions")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("classifier_sentiment_output.png")
    print("Visualization: Classifier predictions saved as classifier_sentiment_output.png")

    # %% [6. Interview Scenario: Classifier-Based Sentiment]
    """
    Interview Scenario: Classifier-Based Sentiment
    Q: What are the advantages of classifier-based sentiment analysis?
    A: ML classifiers (e.g., Naive Bayes, SVM) learn from data, handling complex patterns.
    Key: More accurate than rule-based but requires labeled data.
    Example: NaiveBayesClassifier.train([(word_feats(tokens), label), ...])
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_classifier_sentiment_demo()