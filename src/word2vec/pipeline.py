import numpy as np
from pathlib import Path
from typing import List, Tuple
from .vocab import Vocab
from .model import SGNSModel

class Word2VecPipeline:
    """A high-level wrapper combining the vocabulary and the trained model."""
    
    def __init__(self, vocab: Vocab, model: SGNSModel):
        self.vocab = vocab
        self.model = model

    def save(self, model_dir: Path):
        print(f"Saving pipeline to {model_dir}...")
        self.vocab.save(model_dir)
        self.model.save(model_dir)

    @classmethod
    def load(cls, model_dir: Path):
        print(f"Loading pipeline from {model_dir}...")
        vocab = Vocab.load(model_dir)
        model = SGNSModel.load(model_dir)
        return cls(vocab, model)

    def get_vector(self, word: str) -> np.ndarray:
        """Returns the embedding vector for a given word."""
        if word not in self.vocab.word2id:
            raise KeyError(f"Word '{word}' not in vocabulary.")
        word_id = self.vocab.word2id[word]
        # In SGNS, the input weights (W_in) are the target word embeddings
        return self.model.W_in[word_id]

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Finds the most similar words using cosine similarity."""
        try:
            query_vec = self.get_vector(word)
        except KeyError:
            return []

        # Calculate cosine similarity with all words
        # dot(A, B) / (norm(A) * norm(B))
        embeddings = self.model.W_in
        dot_products = np.dot(embeddings, query_vec)
        
        norms_embeddings = np.linalg.norm(embeddings, axis=1)
        norm_query = np.linalg.norm(query_vec)
        
        similarities = dot_products / (norms_embeddings * norm_query + 1e-10)
        
        # Get top K indices (excluding the query word itself)
        query_id = self.vocab.word2id[word]
        similarities[query_id] = -np.inf # Ignore the word itself
        
        # argsort sorts ascending, so we take the last 'top_k' and reverse
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(self.vocab.id2word[idx], float(similarities[idx])) for idx in top_indices]