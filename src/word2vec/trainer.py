from .dataset import Word2VecDataset
from .model import SGNSModel
from .optim import SGD

class Trainer:
    def __init__(self, dataset: Word2VecDataset, model: SGNSModel, optimizer: SGD):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer

    def train(self, epochs: int = 5):
        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            
            for centers, contexts, negatives in self.dataset.generate_batches():
                # 1. Forward Pass
                loss = self.model.forward(centers, contexts, negatives)
                total_loss += loss
                batches += 1
                
                # 2. Backward Pass
                grads = self.model.backward()
                
                # 3. Optimize
                self.optimizer.step(grads)
                
                if batches % 1000 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batches} | Loss: {total_loss/batches:.4f}")
            
            print(f"--- Epoch {epoch+1} Complete | Average Loss: {total_loss/batches:.4f} ---")