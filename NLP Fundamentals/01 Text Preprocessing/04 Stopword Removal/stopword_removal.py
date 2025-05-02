# %% [1. Introduction to Stopword Removal]
# Learn stopword removal with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader stopwords punkt
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter

def run_stopword_removal_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop is great! I love the fast processor. The screen is vibrant.
    However, the battery life could be better. Overall, a solid purchase.
    """
    word_tokens = nltk.word_tokenize(text)
    print("Synthetic Text: Retail product review created")
    print(f"Word Tokens (Sample): {word_tokens[:10]}...")

    # %% [3. NLTK Stopword Removal]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in word_tokens if word.lower() not in stop_words]
    print(f"Stopword Removal: {len(word_tokens)} tokens reduced to {len(filtered_tokens)}")
    print(f"Filtered Tokens (Sample): {filtered_tokens[:10]}...")

    # %% [4. Custom Stopwords]
    custom_stopwords = stop_words | {'overall', 'however'}
    custom_filtered = [word for word in word_tokens if word.lower() not in custom_stopwords]
    print(f"Custom Stopword Removal: Reduced to {len(custom_filtered)} tokens")
    print(f"Custom Filtered Tokens (Sample): {custom_filtered[:10]}...")

    # %% [5. Impact on Text Analysis]
    before_freq = Counter(word_tokens)
    after_freq = Counter(filtered_tokens)
    print(f"Stopword Impact: Unique words before={len(before_freq)}, after={len(after_freq)}")

    # %% [6. Visualization]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    top_before = dict(sorted(before_freq.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.bar(top_before.keys(), top_before.values())
    plt.title("Top 5 Words (Before)")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    top_after = dict(sorted(after_freq.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.bar(top_after.keys(), top_after.values())
    plt.title("Top 5 Words (After Stopwords)")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("stopword_removal_output.png")
    print("Visualization: Stopword removal impact saved as stopword_removal_output.png")

    # %% [7. Interview Scenario: Stopword Removal]
    """
    Interview Scenario: Stopword Removal
    Q: When should you remove stopwords in NLP?
    A: Remove stopwords for tasks like text classification to reduce noise, but keep them for tasks like machine translation where context matters.
    Key: Stopwords are common words (e.g., "the", "is") with low semantic value.
    Example: [w for w in word_tokens if w.lower() not in stopwords.words('english')]
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_stopword_removal_demo()