import numpy as np
from typing import Tuple, Dict
from pathlib import Path

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

class SGNSModel:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.W_in = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))
        self.cache = {}

    def forward(self, centers: np.ndarray, contexts: np.ndarray, negatives: np.ndarray) -> float:
        h = self.W_in[centers]                  
        c_pos = self.W_out[contexts]            
        c_neg = self.W_out[negatives]           

        pos_logits = np.sum(h * c_pos, axis=1)  
        neg_logits = np.einsum('bd,bkd->bk', h, c_neg) 

        pos_preds = sigmoid(pos_logits)
        neg_preds = sigmoid(neg_logits)

        self.cache = {
            'centers': centers, 'contexts': contexts, 'negatives': negatives,
            'h': h, 'c_pos': c_pos, 'c_neg': c_neg,
            'pos_preds': pos_preds, 'neg_preds': neg_preds
        }

        pos_loss = -np.log(pos_preds + 1e-7)
        neg_loss = -np.log(1.0 - neg_preds + 1e-7)
        batch_loss = np.mean(pos_loss + np.sum(neg_loss, axis=1))
        
        return float(batch_loss)

    def backward(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        pos_error = self.cache['pos_preds'] - 1.0          
        neg_error = self.cache['neg_preds'] - 0.0          

        h = self.cache['h']
        c_pos = self.cache['c_pos']
        c_neg = self.cache['c_neg']

        grad_out_pos = pos_error[:, np.newaxis] * h        
        grad_out_neg = neg_error[:, :, np.newaxis] * h[:, np.newaxis, :] 

        grad_in = pos_error[:, np.newaxis] * c_pos + np.einsum('bk,bkd->bd', neg_error, c_neg)

        return {
            'in': (self.cache['centers'], grad_in),
            'out_pos': (self.cache['contexts'], grad_out_pos),
            'out_neg': (self.cache['negatives'], grad_out_neg)
        }

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "W_in.npy", self.W_in)
        np.save(output_dir / "W_out.npy", self.W_out)

    @classmethod
    def load(cls, input_dir: Path):
        W_in = np.load(input_dir / "W_in.npy")
        W_out = np.load(input_dir / "W_out.npy")
        
        vocab_size, embed_dim = W_in.shape
        model = cls(vocab_size, embed_dim)
        model.W_in = W_in
        model.W_out = W_out
        return model