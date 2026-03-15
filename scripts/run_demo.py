from word2vec import Vocab, Word2VecDataset, SGNSModel, SGD, Trainer, Word2VecPipeline
from word2vec.config import MODELS_DIR, DATASET_SIZES
from word2vec.utils import download_text8, plot_word_embeddings_3d
import argparse

def run_experiment(args):
    
    print(f"\n{'='*50}")
    print(f"🚀 Starting Word2Vec Experiment (Config: {args.size.upper()})")
    print(f"{'='*50}\n")

    run_dir = MODELS_DIR / f"run_{args.size}" # e.g. "run_small", "run_medium", "run_big"
    
    # --- PHASE 1: Data Preparation ---
    data_path = download_text8()
    
    print("\n[1/4] Loading Data...")
    with open(data_path, 'r') as f:
        limit = DATASET_SIZES[args.size]
        if limit:
            text_data = f.read(limit * 5).split()[:limit] # Read chunk, split, and exact slice
        else:
            text_data = f.read().split()
            
    print(f"Loaded {len(text_data)} tokens.")

    # --- PHASE 2: Vocabulary & Dataset ---
    print("\n[2/4] Building Vocabulary & Dataset...")
    vocab = Vocab(max_size=args.vocab_size)
    corpus_ids = vocab.build(text_data)
    dataset = Word2VecDataset(corpus_ids, vocab, window_size=args.window, num_negatives=args.negatives, batch_size=args.batch_size)

    # --- PHASE 3: Training (or Loading) ---
    print("\n[3/4] Initializing Model...")
    
    if run_dir.exists():
        print(f"Found existing trained model at {run_dir}. Loading it to save time!")
        pipeline = Word2VecPipeline.load(run_dir)
    else:
        print("No cached model found. Starting training loop...")
        model = SGNSModel(vocab_size=len(vocab.word2id), embed_dim=args.embed_dim) 
        optimizer = SGD(model, lr=args.lr)
        
        trainer = Trainer(dataset, model, optimizer)
        trainer.train(epochs=args.epochs)
        
        pipeline = Word2VecPipeline(vocab, model)
        pipeline.save(run_dir)

    # --- PHASE 4: Evaluation & Visualization ---
    print("\n[4/4] Evaluating Embeddings...")
    
    test_words = ["man", "woman", "king", "queen", "apple", "orange", "water", "ice", "paris", "berlin", "france", "germany", "watermelon"]
    
    # Print similarities
    print(f"\nMost similar to 'orange':")
    for word, score in pipeline.most_similar("orange", top_k=5):
        print(f"  - {word}: {score:.4f}")

    # Pop up the 3D Plot!
    print("\nGenerating 3D PCA visualization...")
    plot_word_embeddings_3d(pipeline, test_words)
    
    print("\n✨ Experiment Complete! ✨")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Word2Vec training experiment.")
    
    # Define arguments with default values
    parser.add_argument("--size", type=str, default="medium", help="Dataset size preset (e.g., small, medium, big).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--window", type=int, default=2, help="Context window size.")
    parser.add_argument("--negatives", type=int, default=5, help="Number of negative samples per positive sample.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Maximum vocabulary size.")
    parser.add_argument("--embed-dim", type=int, default=50, help="Dimensionality of word embeddings.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for SGD.")
    
    args = parser.parse_args()
    
    run_experiment(args)