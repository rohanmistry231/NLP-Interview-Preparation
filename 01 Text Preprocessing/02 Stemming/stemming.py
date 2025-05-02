# %% [1. Introduction to Stemming]
# Learn Porter and Snowball stemming with NLTK.

# Setup: pip install nltk numpy matplotlib
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
import matplotlib.pyplot as plt
from collections import Counter

def run_stemming_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop is great! I love the fast processor. The screen is vibrant.
    However, the battery life could be better. Overall, a solid purchase.
    """
    word_tokens = nltk.word_tokenize(text)
    print("Synthetic Text: Retail product review created")
    print(f"Word Tokens (Sample): {word_tokens[:10]}...")

    # %% [3. Porter Stemmer]
    porter = PorterStemmer()
    porter_stems = [porter.stem(word) for word in word_tokens]
    print(f"Porter Stemmer: Stemmed {len(porter_stems)} tokens")
    print(f"Porter Stems (Sample): {porter_stems[:10]}...")

    # %% [4. Snowball Stemmer]
    snowball = SnowballStemmer("english")
    snowball_stems = [snowball.stem(word) for word in word_tokens]
    print(f"Snowball Stemmer: Stemmed {len(snowball_stems)} tokens")
    print(f"Snowball Stems (Sample): {snowball_stems[:10]}...")

    # %% [5. Stemming Comparison]
    diff = [(w, p, s) for w, p, s in zip(word_tokens, porter_stems, snowball_stems) if p != s]
    print(f"Stemming Differences: {len(diff)} tokens differ")
    print(f"Differences (Sample): {diff[:5]}..." if diff else "No differences")

    # %% [6. Visualization]
    porter_freq = Counter(porter_stems)
    snowball_freq = Counter(snowball_stems)
    top_porter = dict(sorted(porter_freq.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(top_porter.keys(), top_porter.values())
    plt.title("Porter Stemmer: Top 5 Stems")
    plt.xlabel("Stems")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    top_snowball = dict(sorted(snowball_freq.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.bar(top_snowball.keys(), top_snowball.values())
    plt.title("Snowball Stemmer: Top 5 Stems")
    plt.xlabel("Stems")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("stemming_output.png")
    print("Visualization: Stemming frequency plots saved as stemming_output.png")

    # %% [7. Interview Scenario: Stemming]
    """
    Interview Scenario: Stemming
    Q: What is the difference between Porter and Snowball stemmers?
    A: Porter is older, simpler; Snowball is more aggressive, supports multiple languages.
    Key: Stemming reduces words to roots but may lose meaning (e.g., "running" → "run").
    Example: PorterStemmer().stem("running")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_stemming_demo()