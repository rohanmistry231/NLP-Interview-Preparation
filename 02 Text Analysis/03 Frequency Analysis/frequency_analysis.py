# %% [1. Introduction to Frequency Analysis]
# Learn word and n-gram frequency analysis with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import ngrams

def run_frequency_analysis_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop from TechCorp is great! I love the fast processor. The screen is vibrant.
    However, the battery life could be better. Overall, a solid purchase from TechCorp.
    """
    word_tokens = nltk.word_tokenize(text)
    print("Synthetic Text: Retail product review created")
    print(f"Word Tokens (Sample): {word_tokens[:10]}...")

    # %% [3. Word Frequency]
    word_freq = Counter(word_tokens)
    print(f"Word Frequency: {len(word_freq)} unique words")
    print(f"Top 5 Words: {word_freq.most_common(5)}")

    # %% [4. N-Gram Frequency]
    bigrams = list(ngrams(word_tokens, 2))
    bigram_freq = Counter(bigrams)
    print(f"Bigram Frequency: {len(bigram_freq)} unique bigrams")
    print(f"Top 5 Bigrams: {bigram_freq.most_common(5)}")

    # %% [5. Visualization]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    top_words = dict(word_freq.most_common(5))
    plt.bar([w for w in top_words.keys()], top_words.values())
    plt.title("Top 5 Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    top_bigrams = dict(bigram_freq.most_common(5))
    plt.bar([' '.join(bg) for bg in top_bigrams.keys()], top_bigrams.values())
    plt.title("Top 5 Bigrams")
    plt.xlabel("Bigrams")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("frequency_analysis_output.png")
    print("Visualization: Word and bigram frequency plots saved as frequency_analysis_output.png")

    # %% [6. Interview Scenario: Frequency Analysis]
    """
    Interview Scenario: Frequency Analysis
    Q: What are n-grams, and how are they used in NLP?
    A: N-grams are sequences of n words, used to capture context and patterns in text.
    Key: Useful for language modeling, text classification; bigrams capture word pairs.
    Example: list(nltk.ngrams(word_tokens, 2))
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_frequency_analysis_demo()