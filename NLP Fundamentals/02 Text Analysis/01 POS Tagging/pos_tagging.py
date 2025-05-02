# %% [1. Introduction to POS Tagging]
# Learn part-of-speech tagging with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt averaged_perceptron_tagger
import nltk
import matplotlib.pyplot as plt
from collections import Counter

def run_pos_tagging_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop from TechCorp is great! I love the fast processor. The screen is vibrant.
    However, the battery life could be better. Overall, a solid purchase from TechCorp.
    """
    word_tokens = nltk.word_tokenize(text)
    print("Synthetic Text: Retail product review created")
    print(f"Word Tokens (Sample): {word_tokens[:10]}...")

    # %% [3. NLTK POS Tagging]
    pos_tags = nltk.pos_tag(word_tokens)
    print(f"POS Tagging: Tagged {len(pos_tags)} tokens")
    print(f"POS Tags (Sample): {pos_tags[:10]}...")

    # %% [4. Custom POS Analysis]
    tag_counts = Counter(tag for word, tag in pos_tags)
    print(f"POS Tag Distribution: {dict(tag_counts)}")

    # %% [5. Visualization]
    top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.figure(figsize=(8, 4))
    plt.bar(top_tags.keys(), top_tags.values())
    plt.title("Top 5 POS Tags")
    plt.xlabel("POS Tags")
    plt.ylabel("Frequency")
    plt.savefig("pos_tagging_output.png")
    print("Visualization: POS tag frequency plot saved as pos_tagging_output.png")

    # %% [6. Interview Scenario: POS Tagging]
    """
    Interview Scenario: POS Tagging
    Q: What is POS tagging, and why is it useful in NLP?
    A: POS tagging assigns grammatical roles (e.g., noun, verb) to words, aiding tasks like lemmatization and NER.
    Key: Improves text understanding and feature extraction for models.
    Example: nltk.pos_tag(nltk.word_tokenize("This laptop is great"))
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    run_pos_tagging_demo()