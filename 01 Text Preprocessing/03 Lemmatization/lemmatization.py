# %% [1. Introduction to Lemmatization]
# Learn lemmatization with NLTK's WordNetLemmatizer.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from collections import Counter

def get_wordnet_pos(word):
    """Map POS tag to WordNet POS."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def run_lemmatization_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop is great! I love the fast processor. The screen is vibrant.
    However, the battery life could be better. Overall, a solid purchase.
    """
    word_tokens = nltk.word_tokenize(text)
    print("Synthetic Text: Retail product review created")
    print(f"Word Tokens (Sample): {word_tokens[:10]}...")

    # %% [3. Basic Lemmatization]
    lemmatizer = WordNetLemmatizer()
    basic_lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
    print(f"Basic Lemmatization: Lemmatized {len(basic_lemmas)} tokens")
    print(f"Basic Lemmas (Sample): {basic_lemmas[:10]}...")

    # %% [4. POS-Aware Lemmatization]
    pos_lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in word_tokens]
    print(f"POS-Aware Lemmatization: Lemmatized {len(pos_lemmas)} tokens")
    print(f"POS-Aware Lemmas (Sample): {pos_lemmas[:10]}...")

    # %% [5. Lemmatization vs. Stemming (Comparison)]
    porter = nltk.PorterStemmer()
    stems = [porter.stem(word) for word in word_tokens]
    comparison = [(w, s, l) for w, s, l in zip(word_tokens, stems, pos_lemmas) if s != l]
    print(f"Lemmatization vs. Stemming: {len(comparison)} tokens differ")
    print(f"Differences (Sample): {comparison[:5]}..." if comparison else "No differences")

    # %% [6. Visualization]
    lemma_freq = Counter(pos_lemmas)
    top_lemmas = dict(sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)[:5])
    plt.figure(figsize=(8, 4))
    plt.bar(top_lemmas.keys(), top_lemmas.values())
    plt.title("Top 5 Lemmas (POS-Aware)")
    plt.xlabel("Lemmas")
    plt.ylabel("Frequency")
    plt.savefig("lemmatization_output.png")
    print("Visualization: Lemma frequency plot saved as lemmatization_output.png")

    # %% [7. Interview Scenario: Lemmatization]
    """
    Interview Scenario: Lemmatization
    Q: Why is lemmatization preferred over stemming in some NLP tasks?
    A: Lemmatization produces valid words (e.g., "better" → "good") using linguistic rules, improving interpretability.
    Key: Requires POS tagging for accuracy; slower but more precise than stemming.
    Example: WordNetLemmatizer().lemmatize("running", pos=wordnet.VERB)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    run_lemmatization_demo()