import json
from collections import Counter
from typing import List, Dict
from pathlib import Path

class Vocab:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.word2id: Dict[str, int] = {"<UNK>": 0} # Added UNK token
        self.id2word: Dict[int, str] = {0: "<UNK>"}
        self.word_counts: Dict[int, int] = {0: 0}
        self.total_words = 0

    def build(self, corpus: List[str]) -> List[int]:
        print("Building vocabulary...")
        counts = Counter(corpus)
        # Leave room for <UNK>
        top_words = counts.most_common(self.max_size - 1)
        
        for idx, (word, count) in enumerate(top_words, start=1):
            self.word2id[word] = idx
            self.id2word[idx] = word
            self.word_counts[idx] = count
            self.total_words += count

        # Convert text to IDs, routing unknowns to 0
        corpus_ids = [self.word2id.get(w, 0) for w in corpus]
        
        # Count how many UNKs we actually have
        unk_count = corpus_ids.count(0)
        self.word_counts[0] = unk_count
        self.total_words += unk_count
        
        print(f"Vocab size: {len(self.word2id)} | Corpus size: {len(corpus_ids)}")
        return corpus_ids

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "vocab.json", "w") as f:
            json.dump({
                "word2id": self.word2id,
                "id2word": {str(k): v for k, v in self.id2word.items()}, 
                "word_counts": {str(k): v for k, v in self.word_counts.items()},
                "total_words": self.total_words
            }, f)

    @classmethod
    def load(cls, input_dir: Path):
        with open(input_dir / "vocab.json", "r") as f:
            data = json.load(f)
            
        vocab = cls()
        vocab.word2id = data["word2id"]
        vocab.id2word = {int(k): v for k, v in data["id2word"].items()}
        vocab.word_counts = {int(k): v for k, v in data["word_counts"].items()}
        vocab.total_words = data["total_words"]
        return vocab