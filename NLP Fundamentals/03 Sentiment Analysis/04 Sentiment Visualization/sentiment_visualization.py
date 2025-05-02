# %% [1. Introduction to Sentiment Visualization]
# Learn visualization of sentiment scores with NLTK and Matplotlib.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader vader_lexicon punkt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

def run_sentiment_visualization_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great! I love the fast processor.",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp. Highly recommend!"
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. VADER Sentiment Scores]
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(review) for review in reviews]
    compounds = [score['compound'] for score in scores]
    print("VADER Sentiment: Compound scores calculated")
    print(f"Compound Scores: {[f'{c:.2f}' for c in compounds]}")

    # %% [4. Sentiment Distribution]
    labels = ['Negative', 'Neutral', 'Positive']
    dist = [sum(s['neg'] for s in scores) / len(scores),
            sum(s['neu'] for s in scores) / len(scores),
            sum(s['pos'] for s in scores) / len(scores)]
    print("Sentiment Distribution: Calculated neg/neu/pos averages")

    # %% [5. Visualization]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(reviews) + 1), compounds, color=['green' if c > 0 else 'red' for c in compounds])
    plt.title("Compound Sentiment Scores")
    plt.xlabel("Review")
    plt.ylabel("Compound Score")
    plt.subplot(1, 2, 2)
    plt.pie(dist, labels=labels, autopct='%1.1f%%', colors=['red', 'gray', 'green'])
    plt.title("Sentiment Distribution")
    plt.savefig("sentiment_visualization_output.png")
    print("Visualization: Sentiment scores and distribution saved as sentiment_visualization_output.png")

    # %% [6. Interview Scenario: Sentiment Visualization]
    """
    Interview Scenario: Sentiment Visualization
    Q: How do you visualize sentiment analysis results?
    A: Use bar plots for individual scores and pie charts for sentiment distribution.
    Key: Visualizations highlight trends and patterns for stakeholders.
    Example: plt.bar(range(len(scores)), [s['compound'] for s in scores])
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    run_sentiment_visualization_demo()