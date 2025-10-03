import os
import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import glob
import re
from typing import Union
import pickle

# --- Global BPE Tokenizer for Pickle Compatibility ---
TOKENIZER_DIR = "bpe-tokenizer"
tokenizer_vocab = os.path.join(TOKENIZER_DIR, "vocab.json")
tokenizer_merges = os.path.join(TOKENIZER_DIR, "merges.txt")
bpe_tokenizer = ByteLevelBPETokenizer(vocab=tokenizer_vocab, merges=tokenizer_merges)
def _bpe_tokenize(text: str) -> list[str]: return bpe_tokenizer.encode(text).tokens

def levenshtein_distance(s1: str, s2: str) -> int:
    # (This function is standard and remains unchanged)
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j+1]+1; deletions = current_row[j]+1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class HybridSearchEngine:
    """Combines a fast TF-IDF retriever with a powerful semantic reranker."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Initializing the Hybrid Search Engine...")
        print("Loading indexes... This may take a moment.")
        
        with open('tfidf_vectorizer.pkl', 'rb') as f: self.tfidf_vectorizer = pickle.load(f)
        with open('tfidf_matrix.pkl', 'rb') as f: self.tfidf_matrix = pickle.load(f)
        with open('paragraphs.json', 'r', encoding='utf-8') as f: self.paragraphs = json.load(f)
        
        self.all_embeddings = np.load('embeddings.npy') # Load raw embeddings for fast reranking
        self.semantic_model = SentenceTransformer(model_name)
        
        self.para_ids = list(self.paragraphs.keys())
        self.para_texts = [p['text'] for p in self.paragraphs.values()]
        self.vocabulary = set()
        print("All models and data loaded.")

    def build_autocorrect_vocab(self, corpus_path: str, limit: int = 100):
        # (This function is re-introduced)
        print("\nBuilding auto-correct vocabulary...")
        all_files = glob.glob(os.path.join(corpus_path, "*.txt"))
        if not all_files:
            print("Warning: No local text files found. Auto-correct will be disabled."); return
        np.random.seed(42)
        sample_paths = np.random.choice(all_files, min(limit, len(all_files)), replace=False)
        for file_path in tqdm(sample_paths, desc="Building Vocab"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    words = re.findall(r'\b\w{3,15}\b', f.read().lower())
                    self.vocabulary.update(words)
            except IOError: continue
        print(f"Auto-correct vocabulary built with {len(self.vocabulary)} words.")

    def auto_correct(self, token: str, threshold: int) -> Union[str, None]:
        # (This function is re-introduced)
        if token in self.vocabulary: return token
        min_dist, suggestion = float('inf'), None
        for word in self.vocabulary:
            dist = levenshtein_distance(token, word)
            if dist < min_dist: min_dist, suggestion = dist, word
        return suggestion if min_dist <= threshold else None

    def search(self, query: str, top_k_retrieval: int = 100, top_k_final: int = 10):
        print(f"\n--- Searching for: '{query}' ---")

        if not query or not query.strip():
            print("Query is empty. Please try again.")
            return
        start_time = time.time()
        
        # --- Auto-Correction Step ---
        original_words = query.lower().split()
        corrected_query_list = []
        did_correct = False
        for word in original_words:
            corrected = self.auto_correct(word, 2)
            if corrected and corrected != word:
                print(f"Did you mean: '{corrected}' instead of '{word}'?")
                corrected_query_list.append(corrected); did_correct = True
            else:
                corrected_query_list.append(word)
        if did_correct:
            query = " ".join(corrected_query_list)
            print(f"Performing search with corrected query: '{query}'")

        # --- Stage 1: Fast Retrieval with TF-IDF ---
        query_vector_tfidf = self.tfidf_vectorizer.transform([query])
        cos_scores_tfidf = cosine_similarity(query_vector_tfidf, self.tfidf_matrix).flatten()
        retrieved_indices = cos_scores_tfidf.argsort()[-top_k_retrieval:][::-1]

        # --- Stage 2: Smart Reranking (Now Fast!) ---
        # Get the pre-computed embeddings for only the retrieved candidates
        candidate_embeddings = self.all_embeddings[retrieved_indices]
        
        # Encode the query just once
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        
        # Calculate semantic similarity against the small candidate set
        cos_scores_semantic = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0].cpu().numpy()
        
        # Rerank and get final top results
        reranked_indices_of_candidates = cos_scores_semantic.argsort()[-top_k_final:][::-1]
        final_indices = [retrieved_indices[i] for i in reranked_indices_of_candidates]
        final_scores = [cos_scores_semantic[i] for i in reranked_indices_of_candidates]

        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.4f} seconds.")

        # --- Display Results ---
        if len(final_indices) == 0 or final_scores[0] < 0.3:
            print("\nNo relevant paragraphs found for this query."); return

        print(f"\n--- Top {len(final_indices)} Search Results ---")
        for i, idx in enumerate(final_indices):
            score = final_scores[i]
            if score < 0.3: continue
            
            para_id = self.para_ids[idx]
            info = self.paragraphs[para_id]
            text_snippet = info['text'].replace('\n', ' ').strip()
            snippet = text_snippet[:250] + '...' if len(text_snippet) > 250 else text_snippet
            
            print(f"\n[{para_id}] Score: {score:.4f} | Book: {info['book_title']}")
            print(f"   Snippet: {snippet}")

    def run(self):
        print("\n--- Hybrid Search Engine Ready ---")
        print("Enter your search query. Type 'exit' or 'quit' to end.")
        while True:
            query = input("\nEnter search query: ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting search engine. Goodbye!"); break
            self.search(query)

if __name__ == '__main__':
    CORPUS_PATH = os.path.join('Gutenberg_original','Gutenberg', 'txt')
    engine = HybridSearchEngine()
    engine.build_autocorrect_vocab(corpus_path=CORPUS_PATH, limit=100)
    engine.run()