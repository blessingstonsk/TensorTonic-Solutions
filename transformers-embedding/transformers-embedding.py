import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer with proper initialization.
    """
    embedding = nn.Embedding(vocab_size, d_model)
    
    # Initialize weights (Xavier initialization)
    nn.init.xavier_uniform_(embedding.weight)
    
    return embedding


def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    
    Args:
        embedding: nn.Embedding layer
        tokens: Tensor of shape [seq_len] or [batch_size, seq_len]
        d_model: embedding dimension
    
    Returns:
        Scaled embeddings of shape [..., d_model]
    """
    # Lookup embeddings
    embedded = embedding(tokens)  # shape: [..., d_model]
    
    # Scale embeddings by sqrt(d_model)
    scaled_embeddings = embedded * math.sqrt(d_model)
    
    return scaled_embeddings

