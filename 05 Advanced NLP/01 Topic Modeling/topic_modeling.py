# %% [1. Introduction to Topic Modeling]
# Learn Latent Dirichlet Allocation (LDA) with NLTK and scikit-learn.

# Setup: pip install nltk numpy matplotlib scikit-learn
# NLTK Data: python -m nltk.downloader punkt stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

def run_topic_modeling_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop is great! I love the fast processor and vibrant screen.",
        "The battery life is terrible, but the screen is vibrant.",
        "Solid purchase, fast processor, highly recommend TechCorp.",
        "Poor performance, bad battery, not worth it.",
        "Great laptop, vibrant screen, fast and reliable."
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. Text Preprocessing and Vectorization]
    stop_words = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words=stop_words, lowercase=True)
    X = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vectorization: Matrix shape {X.shape}")

    # %% [4. LDA Topic Modeling]
    n_topics = 2
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    print(f"LDA: {n_topics} topics extracted")

    # %% [5. Topic Extraction]
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-5:]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    # %% [6. Visualization]
    topic_dist = lda.transform(X)
    plt.figure(figsize=(8, 4))
    for i in range(n_topics):
        plt.plot(topic_dist[:, i], label=f'Topic {i+1}')
    plt.title("Topic Distribution Across Reviews")
    plt.xlabel("Review Index")
    plt.ylabel("Topic Probability")
    plt.legend()
    plt.savefig("topic_modeling_output.png")
    print("Visualization: Topic distribution saved as topic_modeling_output.png")

    # %% [7. Interview Scenario: Topic Modeling]
    """
    Interview Scenario: Topic Modeling
    Q: How does LDA work for topic modeling?
    A: LDA assumes documents are mixtures of topics, and topics are distributions over words.
    Key: Uses probabilistic modeling to uncover latent themes.
    Example: LatentDirichletAllocation().fit(X)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_topic_modeling_demo()