# %% [1. Introduction to Tokenization]
# Learn word and sentence tokenization with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt
import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def run_tokenization_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop is great! I love the fast processor. The screen is vibrant.
    However, the battery life could be better. Overall, a solid purchase.
    """
    print("Synthetic Text: Retail product review created")
    print(f"Original Text:\n{text.strip()}")

    # %% [3. Word Tokenization]
    word_tokens = nltk.word_tokenize(text)
    print(f"Word Tokenization: {len(word_tokens)} tokens")
    print(f"Word Tokens: {word_tokens[:10]}...")

    # %% [4. Sentence Tokenization]
    sentence_tokens = nltk.sent_tokenize(text)
    print(f"Sentence Tokenization: {len(sentence_tokens)} sentences")
    print(f"Sentence Tokens: {sentence_tokens}")

    # %% [5. Token Visualization]
    word_freq = Counter(word_tokens)
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.figure(figsize=(8, 4))
    plt.bar(top_words.keys(), top_words.values())
    plt.title("Top 5 Word Frequencies")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.savefig("tokenization_output.png")
    print("Visualization: Word frequency plot saved as tokenization_output.png")

    # %% [6. Interview Scenario: Tokenization]
    """
    Interview Scenario: Tokenization
    Q: Why is tokenization important in NLP?
    A: Tokenization breaks text into meaningful units (words, sentences) for analysis.
    Key: Enables downstream tasks like POS tagging, classification; affects model performance.
    Example: nltk.word_tokenize("This laptop is great!")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_tokenization_demo()