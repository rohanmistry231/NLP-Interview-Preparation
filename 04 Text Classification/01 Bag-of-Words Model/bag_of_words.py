# %% [1. Introduction to Bag-of-Words Model]
# Learn text vectorization with Bag-of-Words using scikit-learn.

# Setup: pip install nltk numpy matplotlib scikit-learn
# NLTK Data: python -m nltk.downloader punkt stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

def run_bag_of_words_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great! I love the fast processor.",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp. Highly recommend!"
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. Text Vectorization]
    stop_words = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words=stop_words, lowercase=True)
    bow_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Bag-of-Words: Matrix shape {bow_matrix.shape}")
    print(f"Feature Names (Sample): {feature_names[:10]}...")

    # %% [4. BoW Feature Extraction]
    bow_array = bow_matrix.toarray()
    print("BoW Matrix (Sample):")
    for i, review in enumerate(reviews[:2]):
        print(f"Review {i+1}: {bow_array[i][:10]}...")

    # %% [5. Visualization]
    word_sums = np.sum(bow_array, axis=0)
    top_indices = np.argsort(word_sums)[-5:]
    top_words = [feature_names[i] for i in top_indices]
    top_counts = [word_sums[i] for i in top_indices]
    plt.figure(figsize=(8, 4))
    plt.bar(top_words, top_counts, color='blue')
    plt.title("Top 5 Words in Bag-of-Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.savefig("bag_of_words_output.png")
    print("Visualization: Top words saved as bag_of_words_output.png")

    # %% [6. Interview Scenario: Bag-of-Words]
    """
    Interview Scenario: Bag-of-Words Model
    Q: What is the Bag-of-Words model, and what are its limitations?
    A: BoW represents text as word frequency vectors, ignoring word order.
    Key: Simple but loses context and semantics.
    Example: CountVectorizer().fit_transform(reviews)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_bag_of_words_demo()