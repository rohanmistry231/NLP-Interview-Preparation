# %% [1. Introduction to Evaluation Metrics]
# Learn accuracy and F1-score for text classification.

# Setup: pip install nltk numpy matplotlib scikit-learn
# NLTK Data: python -m nltk.downloader punkt stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def run_evaluation_metrics_demo():
    # %% [2. Synthetic Retail Text Data]
    train_reviews = [
        ("This laptop is great! I love the fast processor.", "positive"),
        ("The screen is vibrant and solid.", "positive"),
        ("The battery life is terrible.", "negative"),
        ("Poor performance, bad purchase.", "negative")
    ]
    test_reviews = [
        ("This laptop from TechCorp is great!", "positive"),
        ("The battery life is terrible.", "negative"),
        ("A solid purchase from TechCorp.", "positive")
    ]
    train_texts, train_labels = zip(*train_reviews)
    test_texts, test_labels = zip(*test_reviews)
    print("Synthetic Text: Retail product reviews created")
    print(f"Training Data: {len(train_reviews)} labeled reviews")
    print(f"Test Data: {len(test_reviews)} labeled reviews")

    # %% [3. TF-IDF Vectorization]
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"TF-IDF: Training matrix shape {X_train.shape}, Test matrix shape {X_test.shape}")

    # %% [4. Train and Predict]
    classifier = MultinomialNB()
    classifier.fit(X_train, train_labels)
    predictions = classifier.predict(X_test)
    print("Naive Bayes: Predictions made")
    print(f"Predictions: {list(predictions)}")

    # %% [5. Evaluation Metrics]
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    print(f"Evaluation: Accuracy={accuracy:.2f}, F1-Score={f1:.2f}")

    # %% [6. Visualization]
    metrics = ['Accuracy', 'F1-Score']
    values = [accuracy, f1]
    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['blue', 'green'])
    plt.title("Classifier Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig("evaluation_metrics_output.png")
    print("Visualization: Performance metrics saved as evaluation_metrics_output.png")

    # %% [7. Interview Scenario: Evaluation Metrics]
    """
    Interview Scenario: Evaluation Metrics
    Q: Why use F1-score instead of accuracy for text classification?
    A: F1-score balances precision and recall, better for imbalanced datasets.
    Key: Accuracy can be misleading when classes are uneven.
    Example: f1_score(y_true, y_pred, average='weighted')
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_evaluation_metrics_demo()