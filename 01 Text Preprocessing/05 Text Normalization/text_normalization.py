# %% [1. Introduction to Text Normalization]
# Learn text normalization (case, punctuation, whitespace) with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt
import nltk
import string
import re
import matplotlib.pyplot as plt
from collections import Counter

def run_text_normalization_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This LAPTOP is Great!!! I LOVE the Fast Processor... The SCREEN is Vibrant.
    However,,, the battery Life could be better. Overall, a SOLID purchase!!
    """
    print("Synthetic Text: Retail product review created")
    print(f"Original Text:\n{text.strip()}")

    # %% [3. Case Normalization]
    case_normalized = text.lower()
    print("Case Normalization: Converted text to lowercase")
    print(f"Case Normalized (Sample): {case_normalized[:50]}...")

    # %% [4. Punctuation Removal]
    no_punctuation = case_normalized.translate(str.maketrans("", "", string.punctuation))
    print("Punctuation Removal: Removed all punctuation")
    print(f"No Punctuation (Sample): {no_punctuation[:50]}...")

    # %% [5. Whitespace Handling]
    normalized_text = re.sub(r'\s+', ' ', no_punctuation).strip()
    print("Whitespace Handling: Standardized whitespace")
    print(f"Normalized Text (Sample): {normalized_text[:50]}...")

    # %% [6. Tokenization and Visualization]
    word_tokens = nltk.word_tokenize(normalized_text)
    word_freq = Counter(word_tokens)
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.figure(figsize=(8, 4))
    plt.bar(top_words.keys(), top_words.values())
    plt.title("Top 5 Words (Normalized)")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.savefig("text_normalization_output.png")
    print("Visualization: Normalized word frequency plot saved as text_normalization_output.png")

    # %% [7. Interview Scenario: Text Normalization]
    """
    Interview Scenario: Text Normalization
    Q: How does text normalization affect NLP performance?
    A: Normalization (e.g., lowercase, punctuation removal) reduces variability, improving model consistency.
    Key: Essential for tasks like text classification, but over-normalization may lose context.
    Example: text.lower().translate(str.maketrans("", "", string.punctuation))
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_text_normalization_demo()