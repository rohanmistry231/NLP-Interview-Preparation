# %% [1. Introduction to VADER Sentiment Analyzer]
# Learn sentiment analysis with NLTK's VADER.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader vader_lexicon punkt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

def run_vader_sentiment_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great! I love the fast processor.",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp. Highly recommend!"
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. VADER Sentiment Scoring]
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(review) for review in reviews]
    print("VADER Sentiment: Compound scores calculated")
    for i, score in enumerate(scores):
        print(f"Review {i+1}: {score['compound']:.2f} (neg={score['neg']:.2f}, neu={score['neu']:.2f}, pos={score['pos']:.2f})")

    # %% [4. Visualization]
    compounds = [score['compound'] for score in scores]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(reviews) + 1), compounds, color=['green' if c > 0 else 'red' for c in compounds])
    plt.title("VADER Compound Sentiment Scores")
    plt.xlabel("Review")
    plt.ylabel("Compound Score")
    plt.savefig("vader_sentiment_output.png")
    print("Visualization: Compound scores saved as vader_sentiment_output.png")

    # %% [5. Interview Scenario: VADER Sentiment]
    """
    Interview Scenario: VADER Sentiment Analyzer
    Q: How does VADER calculate sentiment scores?
    A: VADER uses a lexicon-based approach with rules for negation, punctuation, and emojis.
    Key: Compound score combines neg/neu/pos scores, normalized to [-1, 1].
    Example: SentimentIntensityAnalyzer().polarity_scores("This laptop is great!")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    run_vader_sentiment_demo()