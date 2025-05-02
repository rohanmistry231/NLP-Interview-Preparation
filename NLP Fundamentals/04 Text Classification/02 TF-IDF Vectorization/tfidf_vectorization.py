# %% [1. Introduction to TF-IDF Vectorization]
# Learn TF-IDF vectorization using scikit-learn.

# Setup: pip install nltk numpy matplotlib scikit-learn
# NLTK Data: python -m nltk.downloader punkt stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

def run_tfidf_vectorization_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great! I love the fast processor.",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp. Highly recommend!"
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. TF-IDF Calculation]
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    print(f"TF-IDF: Matrix shape {tfidf_matrix.shape}")
    print(f"Feature Names (Sample): {feature_names[:10]}...")

    # %% [4. Feature Weighting]
    tfidf_array = tfidf_matrix.toarray()
    print("TF-IDF Matrix (Sample):")
    for i, review in enumerate(reviews[:2]):
        print(f"Review {i+1}: {tfidf_array[i][:10].round(2)}...")

    # %% [5. Visualization]
    word_sums = np.sum(tfidf_array, axis=0)
    top_indices = np.argsort(word_sums)[-5:]
    top_words = [feature_names[i] for i in top_indices]
    top_weights = [word_sums[i] for i in top_indices]
    plt.figure(figsize=(8, 4))
    plt.bar(top_words, top_weights, color='green')
    plt.title("Top 5 TF-IDF Weighted Words")
    plt.xlabel("Words")
    plt.ylabel("TF-IDF Weight")
    plt.savefig("tfidf_vectorization_output.png")
    print("Visualization: Top TF-IDF words saved as tfidf_vectorization_output.png")

    # %% [6. Interview Scenario: TF-IDF Vectorization]
    """
    Interview Scenario: TF-IDF Vectorization
    Q: How does TF-IDF improve over Bag-of-Words?
    A: TF-IDF weights words by importance, reducing impact of common words.
    Key: Combines term frequency and inverse document frequency.
    Example: TfidfVectorizer().fit_transform(reviews)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_tfidf_vectorization_demo()