# %% [1. Introduction to Word Embeddings]
# Learn Word2Vec and GloVe integration with NLTK and gensim.

# Setup: pip install nltk numpy matplotlib gensim
# NLTK Data: python -m nltk.downloader punkt stopwords
# Download GloVe: glove.6B.50d.txt from https://nlp.stanford.edu/projects/glove/
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.manifold import TSNE

def run_word_embeddings_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop is great! I love the fast processor and vibrant screen.",
        "The battery life is terrible, but the screen is vibrant.",
        "Solid purchase, fast processor, highly recommend TechCorp.",
        "Poor performance, bad battery, not worth it.",
        "Great laptop, vibrant screen, fast and reliable."
    ]
    sentences = [nltk.word_tokenize(review.lower()) for review in reviews]
    stop_words = set(stopwords.words('english'))
    sentences = [[word for word in sentence if word not in stop_words] for sentence in sentences]
    print("Synthetic Text: Retail product reviews created")
    print(f"Processed Sentences (Sample): {sentences[0][:10]}...")

    # %% [3. Word2Vec Training]
    w2v_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4)
    print("Word2Vec: Model trained")
    print(f"Word2Vec Similarity (great, fast): {w2v_model.wv.similarity('great', 'fast'):.2f}")

    # %% [4. GloVe Integration]
    glove_file = "glove.6B.50d.txt"  # Adjust path if needed
    try:
        glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
        print("GloVe: Model loaded")
        print(f"GloVe Similarity (great, fast): {glove_model.similarity('great', 'fast'):.2f}")
    except FileNotFoundError:
        print("GloVe: File not found, skipping GloVe demo")

    # %% [5. Visualization]
    words = ['great', 'fast', 'vibrant', 'battery', 'screen']
    if 'glove_model' in locals():
        vectors = [glove_model[word] for word in words if word in glove_model]
        model_name = "GloVe"
    else:
        vectors = [w2v_model.wv[word] for word in words]
        model_name = "Word2Vec"
    tsne = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne.fit_transform(np.array(vectors))
    plt.figure(figsize=(8, 4))
    for i, word in enumerate(words):
        if i < len(vectors):
            plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
    plt.title(f"{model_name} Embeddings (TSNE)")
    plt.savefig("word_embeddings_output.png")
    print(f"Visualization: {model_name} embeddings saved as word_embeddings_output.png")

    # %% [6. Interview Scenario: Word Embeddings]
    """
    Interview Scenario: Word Embeddings
    Q: What are the advantages of Word2Vec over GloVe?
    A: Word2Vec is simpler, trains faster on small datasets; GloVe leverages global co-occurrence statistics.
    Key: Both capture semantic relationships but differ in training approach.
    Example: Word2Vec(sentences, vector_size=50).wv.similarity('great', 'fast')
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_word_embeddings_demo()