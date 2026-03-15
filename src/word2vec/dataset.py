import numpy as np
from typing import List, Tuple, Generator
from .vocab import Vocab

class Word2VecDataset:
    def __init__(self, corpus_ids: List[int], vocab: Vocab, window_size: int = 5, 
                 num_negatives: int = 5, batch_size: int = 128, subsample_t: float = 1e-4):
        self.vocab = vocab
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        
        # 1. Subsample the corpus first
        self.corpus = self._subsample(np.array(corpus_ids), subsample_t)
        
        # 2. Initialize negative sampling distribution
        self._init_negative_table()

    def _subsample(self, corpus_array: np.ndarray, t: float) -> np.ndarray:
        """Drops highly frequent words probabilistically to speed up and improve training
        (Model is not wasting as much time on learning associations with common words like a,an,the ....)."""
        print("Subsampling frequent words...")
        
        # Calculate frequency fraction (z) for every word in the vocabulary
        counts = np.array([self.vocab.word_counts.get(i, 0) for i in range(len(self.vocab.word2id))])
        z = counts / self.vocab.total_words
        
        # Map z-values to the actual words in the corpus array
        z_corpus = z[corpus_array]
        
        # Mikolov's keep probability formula
        # Add a tiny epsilon (1e-10) to prevent division by zero for unmapped words
        p_keep = (np.sqrt(z_corpus / t) + 1) * (t / (z_corpus + 1e-10))
        
        # Generate random numbers and create a boolean mask of words to keep
        random_draws = np.random.rand(len(corpus_array))
        keep_mask = random_draws < p_keep
        
        subsampled_corpus = corpus_array[keep_mask]
        
        print(f"Original length: {len(corpus_array)} | Subsampled length: {len(subsampled_corpus)}")
        print(f"Removed {len(corpus_array) - len(subsampled_corpus)} filler words.")
        
        return subsampled_corpus

    def _init_negative_table(self):
        """Creates a probability distribution for negative sampling (frequency ^ 0.75)."""
        counts = np.array([self.vocab.word_counts.get(i, 0) for i in range(len(self.vocab.word2id))])
        pow_counts = counts ** 0.75
        self.neg_probs = pow_counts / np.sum(pow_counts)

    def generate_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Yields batches of (centers, contexts, negatives)."""
        centers, contexts = [], []
        
        # Sliding window over the NOW SUBSAMPLED corpus
        for i, center_id in enumerate(self.corpus):
            # The window size is often randomized in the original C code for better context, 
            # but a fixed window is fine for this implementation to keep things clean.
            start = max(0, i - self.window_size)
            end = min(len(self.corpus), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    centers.append(center_id)
                    contexts.append(self.corpus[j])
                    
            if len(centers) >= self.batch_size:
                batch_centers = np.array(centers[:self.batch_size])
                batch_contexts = np.array(contexts[:self.batch_size])
                
                # Sample negative words efficiently for the whole batch
                batch_negatives = np.random.choice(
                    len(self.vocab.word2id), 
                    size=(self.batch_size, self.num_negatives), 
                    p=self.neg_probs
                )
                
                yield batch_centers, batch_contexts, batch_negatives
                
                # Keep the remainder for the next batch
                centers = centers[self.batch_size:]
                contexts = contexts[self.batch_size:]