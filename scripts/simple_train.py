from word2vec import Vocab, Word2VecDataset, SGNSModel, SGD, Trainer, Word2VecPipeline
from word2vec.config import MODELS_DIR
from word2vec.utils import download_text8

def main():
    print("🚀 Starting Simple Word2Vec Training Pipeline...")

    # --- 1. Load Data ---
    data_path = download_text8()
    
    with open(data_path, 'r') as f:
        # Read roughly 1.5 million words for a quick training run
        text_data = f.read(10_000_000).split()
    print(f"Loaded {len(text_data)} words.")

    # --- 2. Build Vocabulary & Dataset ---
    vocab = Vocab(max_size=10000)
    corpus_ids = vocab.build(text_data)

    dataset = Word2VecDataset(
        corpus_ids, 
        vocab, 
        window_size=2, 
        num_negatives=5, 
        batch_size=128
    )

    # --- 3. Initialize Model & Optimizer ---
    model = SGNSModel(vocab_size=len(vocab.word2id), embed_dim=50) 
    optimizer = SGD(model, lr=0.05)

    # --- 4. Train & Save ---
    trainer = Trainer(dataset, model, optimizer)
    trainer.train(epochs=2)

    print("\n💾 Saving the pipeline...")
    run_dir = MODELS_DIR / "run_simple"
    pipeline = Word2VecPipeline(vocab, model)
    pipeline.save(run_dir)
    
    print(f"✨ Training complete! Model saved to {run_dir}")

if __name__ == "__main__":
    main()