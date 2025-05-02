# %% [1. Introduction to Chatbot Basics]
# Learn a simple rule-based chatbot with NLTK.

# Setup: pip install nltk numpy matplotlib
# NLTK Data: python -m nltk.downloader punkt
import nltk
from nltk.chat.util import Chat, reflections
import matplotlib.pyplot as plt
from collections import Counter

def run_chatbot_basics_demo():
    # %% [2. Synthetic Retail Text Data]
    pairs = [
        [r"hi|hello|hey", ["Hello! Welcome to TechCorp support. How can I help you?"]],
        [r"laptop (.*) great", ["Glad you love the laptop! What's great about it?"]],
        [r"battery (.*) terrible", ["Sorry to hear about the battery. Can you describe the issue?"]],
        [r"screen (.*) vibrant", ["Awesome, the vibrant screen is a fan favorite! Anything else?"]],
        [r"quit|exit", ["Goodbye! Thanks for chatting with TechCorp."]],
        [r"(.*)", ["I'm not sure I understand. Could you clarify?"]]
    ]
    print("Synthetic Text: Retail chatbot patterns created")
    print(f"Chatbot Rules: {len(pairs)} patterns defined")

    # %% [3. Rule-Based Chatbot]
    chatbot = Chat(pairs, reflections)
    print("Chatbot: Initialized with rule-based patterns")

    # %% [4. Chatbot Interaction Demo]
    sample_inputs = [
        "Hello",
        "The laptop is great",
        "Battery life is terrible",
        "Screen is vibrant",
        "Quit"
    ]
    responses = []
    for input_text in sample_inputs:
        response = chatbot.respond(input_text)
        responses.append((input_text, response))
        print(f"User: {input_text}")
        print(f"Chatbot: {response}")

    # %% [5. Visualization]
    response_lengths = [len(nltk.word_tokenize(resp[1])) for resp in responses]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(responses)), response_lengths, color='purple')
    plt.title("Chatbot Response Lengths")
    plt.xlabel("Interaction")
    plt.ylabel("Response Word Count")
    plt.savefig("chatbot_basics_output.png")
    print("Visualization: Response lengths saved as chatbot_basics_output.png")

    # %% [6. Interview Scenario: Chatbot Basics]
    """
    Interview Scenario: Chatbot Basics
    Q: What are the limitations of rule-based chatbots?
    A: Rule-based chatbots rely on predefined patterns, lacking flexibility and context understanding.
    Key: Simple to build but cannot handle complex or unseen inputs.
    Example: nltk.chat.util.Chat(pairs, reflections)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_chatbot_basics_demo()