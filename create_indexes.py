# create_indexes.py
# This script now saves the raw embeddings file needed for fast reranking.

import os
import re
import glob
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from tokenizers import ByteLevelBPETokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import faiss
import random

# (All configurations and helper functions remain the same)
CORPUS_PATH = os.path.join('Gutenberg_original','Gutenberg', 'txt')
NUM_BOOKS_TO_PROCESS = 1000
TOKENIZER_DIR = "bpe-tokenizer"
BPE_VOCAB_SIZE = 10000
MODEL_NAME = 'all-MiniLM-L6-v2'

def _extract_title(text: str) -> str:
    match = re.search(r"^Title:\s*(.*)", text, re.IGNORECASE | re.MULTILINE)
    if match: return match.group(1).strip()
    else:
        lines = text.strip().split('\n')
        for line in lines:
            if line.strip(): return line.strip()[:80]
        return "Unknown Title"

def train_bpe_tokenizer(paths: list, vocab_size: int, output_dir: str):
    print(f"\n--- Training BPE Tokenizer ---")
    tokenizer = ByteLevelBPETokenizer()
    def file_iterator():
        for path in tqdm(paths, desc="Reading files for tokenizer"):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f: yield f.read()
            except IOError: continue
    tokenizer.train_from_iterator(file_iterator(), vocab_size=vocab_size, min_frequency=2, special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    print(f"Tokenizer trained and saved to '{output_dir}'.")

if __name__ == '__main__':
    print("--- Starting Unified Index Creation Process ---")
    all_files = glob.glob(os.path.join(CORPUS_PATH, "*.txt"))
    random.seed(42)
    random.shuffle(all_files)
    book_sample = all_files[:NUM_BOOKS_TO_PROCESS]
    print(f"Processing a random sample of {len(book_sample)} books.")

    if not os.path.exists(TOKENIZER_DIR):
        train_bpe_tokenizer(all_files, BPE_VOCAB_SIZE, TOKENIZER_DIR)
    
    tokenizer_vocab = os.path.join(TOKENIZER_DIR, "vocab.json")
    tokenizer_merges = os.path.join(TOKENIZER_DIR, "merges.txt")
    bpe_tokenizer = ByteLevelBPETokenizer(vocab=tokenizer_vocab, merges=tokenizer_merges)
    def _bpe_tokenize(text: str): return bpe_tokenizer.encode(text).tokens

    paragraphs_data = {}
    all_paragraphs_text = []
    para_id_counter = 0
    for file_path in tqdm(book_sample, desc="Reading Books"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: text = f.read()
        except IOError: continue
        book_title = _extract_title(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if 50 < len(p.strip()) < 2000]
        for para_text in paragraphs:
            para_id = f"p_{para_id_counter}"
            paragraphs_data[para_id] = {'text': para_text, 'book_title': book_title}
            all_paragraphs_text.append(para_text)
            para_id_counter += 1
    
    print(f"\nExtracted {len(all_paragraphs_text)} paragraphs.")
    
    print("\nBuilding TF-IDF index...")
    tfidf_vectorizer = TfidfVectorizer(tokenizer=_bpe_tokenize, lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_paragraphs_text)
    with open('tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tfidf_vectorizer, f)
    with open('tfidf_matrix.pkl', 'wb') as f: pickle.dump(tfidf_matrix, f)
    print("TF-IDF index saved.")

    print("\nBuilding Semantic Embeddings & FAISS index (this may take a while)...")
    semantic_model = SentenceTransformer(MODEL_NAME)
    embeddings = semantic_model.encode(all_paragraphs_text, show_progress_bar=True, convert_to_numpy=True)
    
    # --- NEW: Save the raw embeddings needed for fast reranking ---
    np.save('embeddings.npy', embeddings)
    print("Raw embeddings saved to embeddings.npy.")

    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, 'index.faiss')
    print("FAISS index saved.")

    with open('paragraphs.json', 'w', encoding='utf-8') as f: json.dump(paragraphs_data, f)
    print("Paragraph data saved.")
    
    print("\n--- All indexes created successfully! ---")