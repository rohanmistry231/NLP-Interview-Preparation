# %% [1. Introduction to Deep Learning Integration]
# Learn TensorFlow and PyTorch for text classification with NLTK.

# Setup: pip install nltk numpy matplotlib scikit-learn tensorflow torch
# NLTK Data: python -m nltk.downloader punkt stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np

def run_deep_learning_nlp_demo():
    # %% [2. Synthetic Retail Text Data]
    train_reviews = [
        ("This laptop is great! I love the fast processor.", 1),
        ("The screen is vibrant and solid.", 1),
        ("The battery life is terrible.", 0),
        ("Poor performance, bad purchase.", 0)
    ]
    test_reviews = [
        ("This laptop from TechCorp is great!", 1),
        ("The battery life is terrible.", 0),
        ("A solid purchase from TechCorp.", 1)
    ]
    train_texts, train_labels = zip(*train_reviews)
    test_texts, test_labels = zip(*test_reviews)
    print("Synthetic Text: Retail product reviews created")
    print(f"Training Data: {len(train_reviews)} labeled reviews")
    print(f"Test Data: {len(test_reviews)} labeled reviews")

    # %% [3. TF-IDF Vectorization]
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True, max_features=100)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    print(f"TF-IDF: Training matrix shape {X_train.shape}, Test matrix shape {X_test.shape}")

    # %% [4. TensorFlow Model]
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf_history = tf_model.fit(X_train, y_train, epochs=10, verbose=0)
    tf_predictions = (tf_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print("TensorFlow: Predictions made")
    print(f"TensorFlow Predictions: {tf_predictions}")

    # %% [5. PyTorch Model]
    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x

    pt_model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(pt_model.parameters())
    X_train_torch = torch.FloatTensor(X_train)
    y_train_torch = torch.FloatTensor(y_train).reshape(-1, 1)
    for _ in range(10):
        optimizer.zero_grad()
        outputs = pt_model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
    pt_model.eval()
    with torch.no_grad():
        pt_predictions = (pt_model(torch.FloatTensor(X_test)) > 0.5).int().numpy().flatten()
    print("PyTorch: Predictions made")
    print(f"PyTorch Predictions: {pt_predictions}")

    # %% [6. Visualization]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tf_history.history['accuracy'], label='TensorFlow Accuracy')
    plt.title("TensorFlow Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(['TensorFlow', 'PyTorch'], [np.mean(tf_predictions == y_test), np.mean(pt_predictions == y_test)], color=['blue', 'green'])
    plt.title("Model Test Accuracy")
    plt.ylabel("Accuracy")
    plt.savefig("deep_learning_nlp_output.png")
    print("Visualization: Model performance saved as deep_learning_nlp_output.png")

    # %% [7. Interview Scenario: Deep Learning Integration]
    """
    Interview Scenario: Deep Learning Integration
    Q: How do you integrate deep learning with NLP?
    A: Use frameworks like TensorFlow/PyTorch to build models (e.g., LSTM, CNN) on text features like TF-IDF or embeddings.
    Key: Deep learning captures complex patterns but requires more data and compute.
    Example: tf.keras.Sequential([Dense(16), Dense(1, activation='sigmoid')])
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    run_deep_learning_nlp_demo()