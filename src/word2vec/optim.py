import numpy as np
from .model import SGNSModel

class SGD:
    def __init__(self, model: SGNSModel, lr: float = 0.025):
        self.model = model
        self.lr = lr

    def step(self, grads: dict):
        """Updates weights using np.subtract.at to handle duplicate indices safely."""
        centers, dW_in = grads['in']
        contexts, dW_out_pos = grads['out_pos']
        negatives, dW_out_neg = grads['out_neg']

        # np.subtract.at is crucial here. If a word appears twice in the same batch,
        # standard indexing (W[idx] -= grad) will overwrite it. 
        # subtract.at accumulates the gradients properly.
        np.subtract.at(self.model.W_in, centers, self.lr * dW_in)
        np.subtract.at(self.model.W_out, contexts, self.lr * dW_out_pos)
        np.subtract.at(self.model.W_out, negatives, self.lr * dW_out_neg)