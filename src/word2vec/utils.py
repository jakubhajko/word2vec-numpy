from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List
from .config import RAW_DATA_DIR
from .pipeline import Word2VecPipeline

def download_text8() -> Path:
    """Downloads and extracts the text8 dataset."""
    zip_path = RAW_DATA_DIR / "text8.zip"
    extract_path = RAW_DATA_DIR / "text8"

    if extract_path.exists():
        print(f"Data ready at {extract_path}.")
        return extract_path

    if not zip_path.exists():
        print(f"Downloading text8 dataset to {zip_path}... (this might take a minute, its ~100MB)")
        url = "http://mattmahoney.net/dc/text8.zip"
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
        
    zip_path.unlink()
    return extract_path

def plot_word_embeddings_3d(pipeline: Word2VecPipeline, words: List[str]):
    """Applies PCA to reduce embeddings to 3D and plots them."""
    # 1. Gather the vectors for the requested words
    vectors = []
    valid_words = []
    
    for word in words:
        try:
            vec = pipeline.get_vector(word)
            vectors.append(vec)
            valid_words.append(word)
        except KeyError:
            print(f"Warning: '{word}' is not in the vocabulary. Skipping.")
            
    if len(vectors) < 3:
        print("Not enough valid words to perform 3D PCA.")
        return

    vectors = np.array(vectors)

    # 2. Reduce dimensions using PCA
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    xs = vectors_3d[:, 0]
    ys = vectors_3d[:, 1]
    zs = vectors_3d[:, 2]

    ax.scatter(xs, ys, zs, color='b', s=100)

    # Annotate the points
    for i, word in enumerate(valid_words):
        ax.text(xs[i], ys[i], zs[i], word, size=14, zorder=1, color='k')

    ax.set_title("3D PCA of Word Embeddings")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    
    print("Close the plot window to continue...")
    plt.show()