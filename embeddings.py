import torch
from transformers import AutoTokenizer, AutoModel

# === Model Configuration ===
MODEL_NAME = "intfloat/e5-small-v2"  # 384-dim output, small size

# === Load Model & Tokenizer Once ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Disable dropout, etc.

# === Generate Embedding ===
def get_embedding(text: str) -> list:
    """
    Generate a 384-dimensional vector embedding for the input text.
    
    Args:
        text (str): The input sentence or paragraph.
    
    Returns:
        list: Embedding vector as a list of floats.
    """
    # Preprocess input (per E5 model's requirement)
    text = text.strip()
    if not text.startswith("query:") and not text.startswith("passage:"):
        text = "passage: " + text

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Generate embeddings (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.tolist()
