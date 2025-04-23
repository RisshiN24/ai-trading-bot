# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Pre-trained NLP model
import torch  # Deep learning framework
from typing import Tuple  # Type hinting

# Set device to GPU if available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the FinBERT tokenizer and model for financial sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Define sentiment labels corresponding to model output
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news: list) -> Tuple[float, str]:
    """
    Estimates the sentiment of given financial news headlines.

    Args:
        news (list): A list of news headlines (strings).

    Returns:
        Tuple[float, str]: Probability of the predicted sentiment and its label.
    """
    if news:
        # Tokenize input text and move it to the appropriate device (CPU/GPU)
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        # Perform sentiment classification using the model
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]

        # Apply softmax to get probability distribution and extract the most likely sentiment
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        
        return probability.item(), sentiment  # Convert probability to a Python float
    else:
        return 0.0, labels[-1]  # Default to "neutral" sentiment if input is empty

# Test the function when the script is run directly
if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['Dow Jones Today: S&P 500 Falls Big, People Crying in the Streets'])
    print(tensor, sentiment)  # Print probability and sentiment label
    print("CUDA Available:", torch.cuda.is_available())  # Check if CUDA (GPU) is accessible