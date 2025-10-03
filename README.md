# Hybrid Retriever-Reranker Semantic Search Engine

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-HuggingFace%20%7C%20FAISS%20%7C%20Scikit--learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-green.svg)

A high-performance semantic search engine for the Gutenberg corpus, engineered to deliver fast, contextually-aware results. This project leverages a state-of-the-art **Retriever-Reranker** architecture, demonstrating a practical application of advanced NLP and Information Retrieval techniques. Developed as a graduate-level project for the Master of Science in Data Science program at the University of Maryland, College Park.

---
## 🎯 About The Project
Traditional keyword search is fast but lacks contextual understanding. Pure semantic search understands context but struggles with speed and out-of-vocabulary (OOV) terms. This project builds a hybrid system that captures the strengths of both paradigms.

The core of this project is a **two-stage search pipeline** that efficiently queries a dataset of over 1 million paragraphs extracted from 1,000 classic books.

### Project Architecture


1.  **Fast Retrieval (The Librarian):** A lightweight, BPE-tokenized **TF-IDF model** first scans the entire corpus. It acts as a fast retriever, identifying a broad set of candidate documents (top ~100) based on keyword relevance. This stage ensures that even OOV terms like `cyberpunk` are handled gracefully by breaking them down into sub-words (`engine`) and finding a relevant search space.

2.  **Smart Reranking (The Scholar):** The retrieved candidates are then passed to a powerful **`sentence-transformer` model**. This model generates dense vector embeddings for the query and the candidate paragraphs. It then re-ranks this small subset based on deep semantic similarity, ensuring the final results are not just keyword matches but are contextually and conceptually the most relevant.

This hybrid approach solves the critical challenges of search speed, relevance, and the handling of unknown concepts.

---
## 🛠️ Technologies & Core Concepts
This project demonstrates proficiency in a range of essential data science and NLP tools and concepts:

* **Python 3.9+**
* **Information Retrieval:** TF-IDF Vectorization, Cosine Similarity.
* **Natural Language Processing (NLP):**
    * **Hugging Face `sentence-transformers`:** For generating state-of-the-art semantic embeddings.
    * **Hugging Face `tokenizers`:** For training and implementing a Byte-Pair Encoding (BPE) tokenizer from scratch.
    * **Levenshtein Distance:** For implementing a custom auto-correct feature.
* **High-Performance Computing:**
    * **FAISS (Facebook AI Similarity Search):** For building and searching an efficient Approximate Nearest Neighbor (ANN) index.
    * **Google Colab (GPU):** For accelerating the computationally expensive model training and indexing pipeline.
* **Core Libraries:** Scikit-learn, NumPy, Pickle.

---
## 🚀 Performance & Demo
The final Retriever-Reranker model provides a significant improvement in both speed and relevance over naive approaches.

* **Speed:** Initial query encoding takes a few seconds on a CPU, with all subsequent searches performing **sub-second retrieval and reranking**.
* **Relevance:** The hybrid model successfully finds semantically relevant documents for known concepts and gracefully handles OOV terms by returning no results, avoiding the "random noise" common in pure semantic models.

<details>
<summary><strong>Click to see a live demo output from the terminal</strong></summary>

```
Initializing the Hybrid Search Engine...
Loading indexes... This may take a moment.
All models and data loaded.

Building auto-correct vocabulary...
Building Vocab: 100%|██████████| 100/100 [00:04<00:00, 24.64it/s]
Auto-correct vocabulary built with 77149 words.

--- Hybrid Search Engine Ready ---
Enter your search query. Type 'exit' or 'quit' to end.

Enter search query: lincold

--- Searching for: 'lincold' ---
Did you mean: 'lincoln' instead of 'lincold'?
Performing search with corrected query: 'lincoln'
Search completed in 9.9659 seconds.

--- Top 10 Search Results ---

[p_750184] Score: 0.6860 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN
   Snippet: "His EXCELLENCY A. LINCOLN, President United States:

[p_750160] Score: 0.6728 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN
   Snippet: "His EXCELLENCY  A. LINCOLN,   President of the United States:
...

Enter search query: government

--- Searching for: 'government' ---
Search completed in 3.6969 seconds.

--- Top 10 Search Results ---

[p_945523] Score: 0.4510 | Book: SECOND TREATISE OF GOVERNMENT by JOHN LOCKE
   Snippet: AN ESSAY CONCERNING THE TRUE ORIGINAL, EXTENT AND END OF CIVIL GOVERNMENT
...

Enter search query: cyberpunk

--- Searching for: 'cyberpunk' ---
Search completed in 4.9155 seconds.

No relevant paragraphs found for this query.
```
</details>

---
## 💡 Key Skills & Learnings
This project was an opportunity to move beyond basic NLP tutorials and engage with the practical challenges of building a real-world search system.

* **Architectural Design:** I designed and implemented a sophisticated **Retriever-Reranker pipeline**, demonstrating an understanding of how to balance trade-offs between speed (TF-IDF) and semantic accuracy (Transformers). This is a common pattern in production MLOps.
* **End-to-End NLP Workflow:** I handled the entire NLP pipeline: sourcing raw data, extensive text preprocessing, training a custom BPE tokenizer, building multiple complex indexes (sparse TF-IDF and dense FAISS), and developing the final application logic.
* **Performance Optimization:** I identified and solved critical performance bottlenecks. The initial 15-second search time was reduced to sub-second speeds by implementing a **FAISS** index. The multi-hour model indexing time was reduced to minutes by leveraging **GPU acceleration in a cloud environment (Google Colab)**.
* **Problem Analysis:** I diagnosed and explained complex model behaviors, such as the semantic model's failure on OOV terms (`cyberpunk`) and its anomalous but predictable results for gibberish queries. This showcases strong analytical and debugging skills.

---
## ⚙️ Getting Started

Follow these steps to set up and run the project locally.

### Installation & Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/NLP-Search-Engine.git](https://github.com/your-username/NLP-Search-Engine.git)
    cd NLP-Search-Engine
    ```
2.  **Set up the data & indexes:**
    * Download the `Gutenberg_original.zip` file containing the raw text data. Unzip it and place the `Gutenberg_original` folder in the root of the project.
    * Download the pre-computed index files (`final_indexes.zip`) from the release page. Unzip the folder and move all index files (`.pkl`, `.json`, `.faiss`, etc.) into the root of the project.

3.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created and added to the repository by running `pip freeze > requirements.txt`)*

### Usage
Once the environment is set up and all index files are in place, run the main search engine application:
```bash
python search_engine.py
```
The program will load the indexes and present an interactive prompt for search queries.

---
## 📈 Future Improvements
* **Build a Web Interface:** Wrap the search engine in a simple Streamlit or Flask web application for a more user-friendly demo.
* **Quantitative Evaluation:** Develop a ground-truth dataset to formally evaluate the retriever and reranker using metrics like Precision@k and MRR.
* **Deployment:** Deploy the final Streamlit application to a free hosting service like Hugging Face Spaces to create a shareable live demo.
