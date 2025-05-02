# %% [1. Introduction to Named Entity Recognition]
# Learn NER with NLTK's chunking.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words
import nltk
import matplotlib.pyplot as plt
from collections import Counter

def run_ner_demo():
    # %% [2. Synthetic Retail Text Data]
    text = """
    This laptop from TechCorp is great! I love the fast processor from Intel.
    The screen is vibrant, designed by Samsung. However, the battery life could be better.
    Overall, a solid purchase from TechCorp in New York.
    """
    word_tokens = nltk.word_tokenize(text)
    print("Synthetic Text: Retail product review created")
    print(f"Word Tokens (Sample): {word_tokens[:10]}...")

    # %% [3. NLTK NER with Chunking]
    pos_tags = nltk.pos_tag(word_tokens)
    chunks = nltk.ne_chunk(pos_tags, binary=False)
    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label') and chunk.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
            entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
    print(f"NER: Extracted {len(entities)} entities")
    print(f"Entities: {entities}")

    # %% [4. Entity Visualization]
    entity_counts = Counter(label for label, _ in entities)
    if entity_counts:
        plt.figure(figsize=(8, 4))
        plt.bar(entity_counts.keys(), entity_counts.values())
        plt.title("Entity Type Frequencies")
        plt.xlabel("Entity Types")
        plt.ylabel("Frequency")
        plt.savefig("ner_output.png")
        print("Visualization: Entity frequency plot saved as ner_output.png")
    else:
        print("Visualization: No entities found, skipping plot")

    # %% [5. Interview Scenario: NER]
    """
    Interview Scenario: Named Entity Recognition
    Q: How does NER work in NLTK, and what are its limitations?
    A: NLTK uses POS tagging and chunking to identify entities (e.g., PERSON, ORGANIZATION).
    Key: Limited to predefined categories; less accurate than modern models like spaCy.
    Example: nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    run_ner_demo()