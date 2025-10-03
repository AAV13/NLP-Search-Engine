import os
import re
import glob
import json
import numpy as np
from tqdm import tqdm
import heapq
import time
from typing import Union
from tokenizers import ByteLevelBPETokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def train_bpe_tokenizer(paths: list, vocab_size: int, output_dir: str):
    """Trains a BPE tokenizer on the corpus and saves it."""
    print(f"\n--- Training BPE Tokenizer ---")
    tokenizer = ByteLevelBPETokenizer()
    
    # Create a generator to read files, replacing any UTF-8 errors.
    # This is necessary to handle potential encoding issues in the Gutenberg corpus.
    def file_iterator():
        for path in tqdm(paths, desc="Reading files for tokenizer"):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    yield f.read()
            except IOError:
                print(f"Warning: Could not read file {path}. Skipping.")
                continue

    # Train on the full corpus to build the best possible vocabulary.
    tokenizer.train_from_iterator(
        file_iterator(),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    print(f"Tokenizer trained and saved to '{output_dir}'.")
    return tokenizer

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

class BPESearchEngine:
    """A search engine using a BPE tokenizer and TF-IDF model."""
    def __init__(self, tokenizer_dir: str):
        print("Initializing the BPE Search Engine...")
        
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.txt")
        
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            raise FileNotFoundError("Tokenizer files (vocab.json, merges.txt) not found.")
            
        self.tokenizer = ByteLevelBPETokenizer(vocab=vocab_path, merges=merges_path)
        
        # Define paths for caching the index to speed up subsequent runs.
        self.vectorizer_cache_path = 'tfidf_vectorizer.pkl'
        self.matrix_cache_path = 'tfidf_matrix.pkl'
        self.paragraphs_cache_path = 'paragraphs_bpe.json'
        
        self.paragraphs = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self.vocabulary = set()
        self.inverted_index = {}

    def _extract_title(self, text: str) -> str:
        """Extracts the book title from its text content."""
        # Look for a line starting with "Title:", case-insensitive.
        match = re.search(r"^Title:\s*(.*)", text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        else:
            # If not found, fall back to the first non-empty line.
            lines = text.strip().split('\n')
            for line in lines:
                if line.strip():
                    return line.strip()[:80]
            return "Unknown Title"

    def _bpe_tokenize(self, text: str) -> list[str]:
        """Custom tokenizer method for scikit-learn's TfidfVectorizer."""
        return self.tokenizer.encode(text).tokens

    def _save_to_cache(self):
        """Saves the search index components to disk."""
        print("\nSaving TF-IDF index to cache...")
        with open(self.vectorizer_cache_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(self.matrix_cache_path, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(self.paragraphs_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.paragraphs, f)
        print("Cache saved successfully.")

    def _load_from_cache(self):
        """Loads the pre-computed search index from disk."""
        print("\nLoading TF-IDF index from cache...")
        with open(self.vectorizer_cache_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(self.matrix_cache_path, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        with open(self.paragraphs_cache_path, 'r', encoding='utf-8') as f:
            self.paragraphs = json.load(f)
            
        self.vocabulary = set(self.vectorizer.get_feature_names_out())
        print(f"Cache loaded. Vocabulary size: {len(self.vocabulary)} tokens.")
        return True

    def index_corpus(self, book_paths: list):
        """Processes books to build the search index."""
        print("\n--- Phase 1: Indexing Corpus ---")
        
        para_id_counter = 0
        all_paragraphs_text = []
        for file_path in tqdm(book_paths, desc="Reading Books"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except IOError:
                continue
            
            book_title = self._extract_title(text)
            paragraphs = [p.strip() for p in text.split('\n\n') if 50 < len(p.strip()) < 2000]
            
            for para_text in paragraphs:
                para_id = f"p_{para_id_counter}"
                self.paragraphs[para_id] = {'text': para_text, 'book_title': book_title}
                all_paragraphs_text.append(para_text)
                para_id_counter += 1
        
        print(f"\nRead {len(self.paragraphs)} paragraphs from {len(book_paths)} books.")
        print("Building TF-IDF Matrix and Inverted Index...")
        
        self.vectorizer = TfidfVectorizer(tokenizer=self._bpe_tokenize, lowercase=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(all_paragraphs_text)
        self.vocabulary = set(self.vectorizer.get_feature_names_out())

        # Build the inverted index to satisfy the assignment's explicit requirement.
        # The TF-IDF matrix is used for the actual search ranking.
        feature_names = self.vectorizer.get_feature_names_out()
        for token in feature_names:
            self.inverted_index[token] = []
        
        for para_idx, para_text in enumerate(all_paragraphs_text):
            para_id = f"p_{para_idx}"
            tokens = self._bpe_tokenize(para_text)
            for token in tokens:
                if token in self.inverted_index:
                    self.inverted_index[token].append(para_id)

        print(f"Indexing complete. Vocabulary size: {len(self.vocabulary)} BPE tokens.")

    def auto_correct(self, token: str, threshold: int) -> Union[str, None]:
        """Suggests a spelling correction for a given token."""
        if token in self.vocabulary:
            return token

        min_dist, suggestion = float('inf'), None
        for word in self.vocabulary:
            dist = levenshtein_distance(token, word)
            if dist < min_dist:
                min_dist, suggestion = dist, word
        
        return suggestion if min_dist <= threshold else None

    def search(self, query: str, top_k: int = 10, autocorrect_threshold: int = 2):
        """Searches the index for a given query."""
        print("\n--- Processing Query ---")
        if not query or not query.strip():
            print("Query is empty. Please try again.")
            return

        original_words = query.lower().split()
        corrected_query = []
        did_correct = False
        for word in original_words:
            corrected = self.auto_correct(word, autocorrect_threshold)
            if corrected and corrected != word:
                print(f"Did you mean: '{corrected}' instead of '{word}'?")
                corrected_query.append(corrected)
                did_correct = True
            else:
                corrected_query.append(word)
        
        if did_correct:
            query = " ".join(corrected_query)
            print(f"Performing search with corrected query: '{query}'")

        start_time = time.time()
        
        query_vector = self.vectorizer.transform([query])
        cos_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = heapq.nlargest(top_k, range(len(cos_scores)), key=cos_scores.__getitem__)
        
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.4f} seconds.")

        if not top_indices or cos_scores[top_indices[0]] == 0:
            print("\nNo relevant paragraphs found for this query.")
            return
            
        print(f"\n--- Top {len(top_indices)} Search Results ---")
        all_para_ids = list(self.paragraphs.keys())
        for idx in top_indices:
            score = cos_scores[idx]
            if score == 0: continue
            
            para_id = all_para_ids[idx]
            info = self.paragraphs[para_id]
            text_snippet = info['text'].replace('\n', ' ').strip()
            snippet = text_snippet[:250] + '...' if len(text_snippet) > 250 else text_snippet
            
            print(f"\n[{para_id}] Score: {score:.4f} | Book: {info['book_title']}")
            print(f"   Snippet: {snippet}")

    def run(self):
        """Starts the main user interaction loop."""
        print("\n--- Search Engine Ready ---")
        print("Enter your search query. Type 'exit' or 'quit' to end.")
        while True:
            query = input("\nEnter search query: ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting search engine. Goodbye!")
                break
            self.search(query)

if __name__ == '__main__':
    
    # --- Configuration ---
    CORPUS_PATH = os.path.join('Gutenberg_original', 'Gutenberg', 'txt')
    NUM_BOOKS_TO_PROCESS = 100
    TOKENIZER_DIR = "bpe-tokenizer"
    BPE_VOCAB_SIZE = 10000

    # --- Execution ---
    all_files = glob.glob(os.path.join(CORPUS_PATH, "*.txt"))
    if not all_files:
        print(f"Error: No .txt files found in '{CORPUS_PATH}'. Exiting.")
    else:
        np.random.seed(42)
        book_paths = np.random.choice(all_files, min(NUM_BOOKS_TO_PROCESS, len(all_files)), replace=False).tolist()

        # Train the tokenizer on the full corpus if it hasn't been trained yet.
        if not os.path.exists(TOKENIZER_DIR):
            train_bpe_tokenizer(paths=all_files, vocab_size=BPE_VOCAB_SIZE, output_dir=TOKENIZER_DIR)
        
        engine = BPESearchEngine(TOKENIZER_DIR)
        
        # Load the index from cache if available, otherwise build it.
        if os.path.exists(engine.vectorizer_cache_path):
            engine._load_from_cache()
        else:
            engine.index_corpus(book_paths)
            engine._save_to_cache()
        
        engine.run()
