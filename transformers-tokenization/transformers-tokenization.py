import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Add special tokens with fixed IDs
        special_tokens = [
            self.pad_token,  # 0
            self.unk_token,  # 1
            self.bos_token,  # 2
            self.eos_token   # 3
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        current_id = len(special_tokens)
        
        # Collect unique words from training texts
        unique_words = set()
        for text in texts:
            words = text.strip().split()
            unique_words.update(words)
        
        # Add words to vocab
        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        self.vocab_size = current_id
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.strip().split()
        
        token_ids = []
        for word in words:
            if word in self.word_to_id:
                token_ids.append(self.word_to_id[word])
            else:
                token_ids.append(self.word_to_id[self.unk_token])
        
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for idx in ids:
            if idx in self.id_to_word:
                word = self.id_to_word[idx]
                # Skip special tokens in decoding (optional behavior)
                if word not in {self.pad_token, self.bos_token, self.eos_token}:
                    words.append(word)
            else:
                words.append(self.unk_token)
        
        return " ".join(words)
