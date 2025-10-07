import glob
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm


# --- Config setup ---
CORPUS_DIR = Path("Gutenberg_original") / "Gutenberg" / "txt"
BOOKS_TO_INDEX = 100
TOKENIZER_ASSETS_DIR = Path("bpe_tokenizer_assets")
VOCAB_SIZE = 10000


@dataclass
class Paragraph:
    """Just something simple to store info about each paragraph."""
    doc_id: str
    text: str
    book_title: str


def train_tokenizer(corpus_files: List[str], vocab_size: int, output_dir: Path):
    """
    Train a ByteLevelBPE tokenizer from a bunch of text files.
    This might take a bit if there are a lot of books.
    """
    print("--- Training BPE Tokenizer (this might take a while) ---")
    tokenizer = ByteLevelBPETokenizer()

    def text_iter():
        # Not perfect, but just cycles through files safely.
        for file_path in tqdm(corpus_files, desc="Reading files"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    yield f.read()
            except IOError:
                print(f"Couldn't read {file_path}, skipping it.")
                continue

    tokenizer.train_from_iterator(
        text_iter(),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    output_dir.mkdir(exist_ok=True)
    tokenizer.save_model(str(output_dir))
    print(f"Done! Tokenizer saved in {output_dir}.")


def calculate_levenshtein(s1: str, s2: str) -> int:
    """Basic Levenshtein distance (edit distance) implementation."""
    if len(s1) < len(s2):
        return calculate_levenshtein(s2, s1)
    if not s2:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            subs = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, subs))
        prev_row = curr_row

    return prev_row[-1]


class CorpusSearchEngine:
    """
    TF-IDF based search over a book corpus with auto-correct and tokenizer handling.
    Basically a mini local search engine.
    """
    def __init__(self, tokenizer_dir: Path):
        print("Initializing Corpus Search Engine...")
        vocab_file = tokenizer_dir / "vocab.json"
        merges_file = tokenizer_dir / "merges.txt"

        if not (vocab_file.exists() and merges_file.exists()):
            raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")

        self.tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        self.documents: List[Paragraph] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.vocabulary: Set[str] = set()

        # Cache setup — makes subsequent runs faster
        self.cache_dir = Path("search_index_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.vectorizer_cache = self.cache_dir / "vectorizer.pkl"
        self.matrix_cache = self.cache_dir / "tfidf_matrix.pkl"
        self.docs_cache = self.cache_dir / "documents.json"

    def _get_bpe_tokens(self, text: str) -> List[str]:
        """Just a tiny helper to tokenize text via BPE."""
        return self.tokenizer.encode(text).tokens

    def _extract_title(self, text: str) -> str:
        """Tries to pull the title out of the text file (if present)."""
        match = re.search(r"^Title:\s*(.*)", text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # fallback: first non-empty line (seen this pattern in Gutenberg files)
        for line in text.strip().split("\n"):
            if line.strip():
                return line.strip()[:80]
        return "Unknown Title"

    def build_index(self, book_paths: List[str]):
        """Turns book files into TF-IDF vectors for searching later."""
        print("\n--- Building search index ---")
        doc_count = 0
        all_texts = []

        for book_path in tqdm(book_paths, desc="Indexing books"):
            try:
                full_text = Path(book_path).read_text(encoding="utf-8", errors="ignore")
            except IOError:
                print(f"Skipping unreadable file: {book_path}")
                continue

            book_title = self._extract_title(full_text)
            paras = [p.strip() for p in full_text.split("\n\n") if 50 < len(p.strip()) < 2000]

            for para in paras:
                pid = f"doc_{doc_count}"
                self.documents.append(Paragraph(pid, para, book_title))
                all_texts.append(para)
                doc_count += 1

        print(f"Collected {len(self.documents)} paragraphs from {len(book_paths)} books.")
        print("Creating TF-IDF matrix...")

        self.vectorizer = TfidfVectorizer(tokenizer=self._get_bpe_tokens, lowercase=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        self.vocabulary = set(self.vectorizer.get_feature_names_out())

        print(f"Done indexing! Found {len(self.vocabulary)} unique tokens.")
        self.save_index_to_disk()

    def save_index_to_disk(self):
        """Stores everything locally so we don’t have to rebuild each time."""
        print("\nSaving index to disk cache...")
        with open(self.vectorizer_cache, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.matrix_cache, "wb") as f:
            pickle.dump(self.tfidf_matrix, f)

        docs_as_dict = [{"doc_id": d.doc_id, "text": d.text, "book_title": d.book_title} for d in self.documents]
        with open(self.docs_cache, "w", encoding="utf-8") as f:
            json.dump(docs_as_dict, f, indent=2)
        print("Cache saved successfully.")

    def load_index_from_disk(self) -> bool:
        """Loads previously saved TF-IDF data if it’s already there."""
        if not all(p.exists() for p in [self.vectorizer_cache, self.matrix_cache, self.docs_cache]):
            return False

        print("\nLoading cached index files...")
        with open(self.vectorizer_cache, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.matrix_cache, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(self.docs_cache, "r", encoding="utf-8") as f:
            docs_raw = json.load(f)
            self.documents = [Paragraph(**d) for d in docs_raw]

        self.vocabulary = set(self.vectorizer.get_feature_names_out())
        print(f"Loaded {len(self.documents)} documents from cache.")
        return True

    def find_correction(self, token: str, max_dist: int) -> Optional[str]:
        """Quick fuzzy matching for spelling correction."""
        if token in self.vocabulary:
            return token

        closest = None
        min_d = float("inf")
        for word in self.vocabulary:
            d = calculate_levenshtein(token, word)
            if d < min_d:
                min_d = d
                closest = word
        return closest if min_d <= max_dist else None

    def _process_query(self, query: str) -> str:
        """
        Handles query preprocessing — lowercasing, spelling fix, and BPE fallback.
        """
        tokens = query.lower().split()
        final_tokens = []
        modified = False

        for t in tokens:
            if t in self.vocabulary:
                final_tokens.append(t)
                continue

            correction = self.find_correction(t, max_dist=2)
            if correction and correction != t:
                print(f"Replacing '{t}' → '{correction}'")
                final_tokens.append(correction)
                modified = True
            else:
                # fallback: break down into smaller BPE tokens
                sub = self._get_bpe_tokens(t)
                print(f"'{t}' not found. Falling back to subwords: {sub}")
                final_tokens.extend(sub)
                modified = True

        processed = " ".join(final_tokens)
        if modified:
            print(f"Processed query: {processed}")
        return processed

    def _rank_results(self, query: str, top_k: int) -> List[Dict]:
        """Compute cosine similarity and return top_k matches."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_ids = np.argsort(sims)[-top_k:][::-1]
        results = []
        for i in top_ids:
            s = sims[i]
            if s > 0:
                doc = self.documents[i]
                results.append({"score": s, "document": doc})
        return results

    def search(self, query: str, top_k: int = 10):
        """Public entrypoint for searching text in the corpus."""
        if not query.strip():
            print("Empty query, please type something.")
            return

        t0 = time.time()
        processed = self._process_query(query)
        results = self._rank_results(processed, top_k)
        t1 = time.time()

        print(f"\nSearch took {t1 - t0:.3f}s.")

        if not results:
            print("No matches found.")
            return

        print(f"\n--- Top {len(results)} results ---")
        for r in results:
            d = r["document"]
            snip = d.text.replace("\n", " ").strip()
            if len(snip) > 250:
                snip = snip[:250] + "..."
            print(f"\n[{d.doc_id}] ({d.book_title}) | score={r['score']:.4f}")
            print(f"  {snip}")

    def start_interactive_session(self):
        """Simple REPL for user queries."""
        print("\n--- Search Engine Ready ---")
        print("Type something to search, or 'exit' to quit.")
        while True:
            try:
                q = input("\nSearch Query> ").strip()
                if q.lower() in ("exit", "quit"):
                    print("Alright, bye 👋")
                    break
                self.search(q)
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting...")
                break


def main():
    """Main entry point."""
    all_files = glob.glob(str(CORPUS_DIR / "*.txt"))
    if not all_files:
        print(f"No .txt files in {CORPUS_DIR}. Check that path.")
        return

    if not TOKENIZER_ASSETS_DIR.exists():
        train_tokenizer(all_files, VOCAB_SIZE, TOKENIZER_ASSETS_DIR)

    engine = CorpusSearchEngine(TOKENIZER_ASSETS_DIR)

    if not engine.load_index_from_disk():
        print("No cached index found — rebuilding from scratch...")
        np.random.seed(42)
        sample_books = np.random.choice(all_files, min(BOOKS_TO_INDEX, len(all_files)), replace=False).tolist()
        engine.build_index(sample_books)

    engine.start_interactive_session()


if __name__ == "__main__":
    main()
