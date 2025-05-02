# %% [1. Introduction to Concordance and Collocations]
# Learn concordance and collocation analysis with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt
import nltk
import matplotlib.pyplot as plt
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def run_concordance_collocations_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop from TechCorp is great! I love the fast processor. The screen is vibrant.
    However, the battery life could be better. Overall, a solid purchase from TechCorp.
    The fast processor and vibrant screen make it a great laptop. TechCorp delivers again!
    """
    word_tokens = nltk.word_tokenize(text)
    text_obj = nltk.Text(word_tokens)
    print("Synthetic Text: Retail product review created")
    print(f"Word Tokens (Sample): {word_tokens[:10]}...")

    # %% [3. Concordance Analysis]
    print("Concordance for 'great':")
    text_obj.concordance('great', width=50, lines=3)

    # %% [4. Collocation Detection]
    finder = BigramCollocationFinder.from_words(word_tokens)
    collocations = finder.nbest(BigramAssocMeasures.likelihood_ratio, 5)
    print(f"Collocations: Top 5 bigrams")
    print(f"Collocations: {collocations}")

    # %% [5. Visualization]
    collocation_counts = Counter([' '.join(bg) for bg in collocations])
    plt.figure(figsize=(8, 4))
    plt.bar(collocation_counts.keys(), collocation_counts.values())
    plt.title("Top 5 Collocations")
    plt.xlabel("Collocations")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.savefig("concordance_collocations_output.png")
    print("Visualization: Collocation frequency plot saved as concordance_collocations_output.png")

    # %% [6. Interview Scenario: Concordance and Collocations]
    """
    Interview Scenario: Concordance and Collocations
    Q: What are collocations, and why are they important in NLP?
    A: Collocations are word pairs that frequently co-occur, revealing meaningful associations.
    Key: Useful for understanding context, improving text generation, and feature extraction.
    Example: BigramCollocationFinder.from_words(word_tokens).nbest()
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_concordance_collocations_demo()