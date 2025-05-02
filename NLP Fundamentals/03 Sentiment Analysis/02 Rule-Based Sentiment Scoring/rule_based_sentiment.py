# %% [1. Introduction to Rule-Based Sentiment Scoring]
# Learn custom rule-based sentiment scoring with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt
import nltk
import matplotlib.pyplot as plt

def run_rule_based_sentiment_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great! I love the fast processor.",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp. Highly recommend!"
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. Custom Sentiment Rules]
    positive_words = {'great', 'love', 'fast', 'vibrant', 'solid', 'recommend'}
    negative_words = {'terrible', 'bad', 'poor'}
    def rule_based_score(review):
        tokens = nltk.word_tokenize(review.lower())
        score = sum(1 for token in tokens if token in positive_words) - sum(1 for token in tokens if token in negative_words)
        return score / len(tokens) if tokens else 0
    scores = [rule_based_score(review) for review in reviews]
    print("Rule-Based Sentiment: Scores calculated")
    for i, score in enumerate(scores):
        print(f"Review {i+1}: {score:.2f}")

    # %% [4. Visualization]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(reviews) + 1), scores, color=['green' if s > 0 else 'red' for s in scores])
    plt.title("Rule-Based Sentiment Scores")
    plt.xlabel("Review")
    plt.ylabel("Normalized Score")
    plt.savefig("rule_based_sentiment_output.png")
    print("Visualization: Rule-based scores saved as rule_based_sentiment_output.png")

    # %% [5. Interview Scenario: Rule-Based Sentiment]
    """
    Interview Scenario: Rule-Based Sentiment Scoring
    Q: When would you use rule-based sentiment scoring?
    A: Use for quick prototyping or when labeled data is scarce; relies on predefined word lists.
    Key: Simple but less robust than ML-based methods.
    Example: Score = (positive_words - negative_words) / total_words
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_rule_based_sentiment_demo()