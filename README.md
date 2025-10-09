# Hybrid Retriever-Reranker Semantic Search Engine

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-HuggingFace%20%7C%20FAISS%20%7C%20Scikit--learn-orange.svg)

Dataset Link: https://shibamoulilahiri.github.io/gutenberg_dataset.html

A semantic search engine for the Gutenberg corpus, engineered to deliver fast, contextually-aware results. This project leverages a **Retriever-Reranker** architecture, demonstrating a practical application of advanced NLP and Information Retrieval techniques.

---
## About The Project
A small hybrid search engine that combines keyword and semantic search. This project explores how search engines balance speed and meaning, demonstrating how traditional term-based retrieval and modern semantic techniques can work together.

The core of this project is a **two-stage search pipeline** that efficiently queries a dataset of over 1 million paragraphs extracted from 1000 classic books from the Gutenberg Dataset. The dataset contains 3000 books but for quicker execution time due to limited resources, the code uses 1000 books.

### Project Architecture


1.  **Fast Retrieval (The Librarian):** A lightweight, BPE-tokenized **TF-IDF model** first scans the entire corpus. It acts as a fast retriever, identifying a broad set of candidate documents (top ~100) based on keyword relevance. This stage ensures that even OOV terms like `cyberpunk` are handled by breaking them down into sub-words (`engine`) and finding a relevant search space.

2.  **Smart Reranking (The Scholar):** The retrieved candidates are then passed to a powerful **`sentence-transformer` model**. This model generates dense vector embeddings for the query and the candidate paragraphs. It then re-ranks this small subset based on deep semantic similarity, making sure that the final results are not just keyword matches but are contextually and conceptually the most relevant.

This hybrid approach solves the critical challenges of search speed, relevance, and the handling of unknown concepts.

---
## Technologies & Core Concepts
This project makes use of a range of essential DS & NLP tools and concepts:

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
## 🚀Performance & ▶️Demo
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

Enter search query: cyberpunk

--- Searching for: 'cyberpunk' ---
Search completed in 4.9155 seconds.

No relevant paragraphs found for this query.

Enter search query:

--- Searching for: '' ---
Query is empty. Please try again.

Enter search query: lincold

--- Searching for: 'lincold' ---
Did you mean: 'lincoln' instead of 'lincold'?
Performing search with corrected query: 'lincoln'
Search completed in 9.9659 seconds.

--- Top 10 Search Results ---

[p_750184] Score: 0.6860 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN      
   Snippet: "His EXCELLENCY A. LINCOLN, President United States:

[p_750160] Score: 0.6728 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN      
   Snippet: "His EXCELLENCY   A. LINCOLN,   President of the United States:      

[p_749677] Score: 0.6214 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN      
   Snippet: A. LINCOLN. By the President: WILLIAM H. SEWARD, Secretary of State. 

[p_750247] Score: 0.6214 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN      
   Snippet: A. LINCOLN. By the President: WILLIAM H. SEWARD, Secretary of State. 

[p_136522] Score: 0.5695 | Book: PERSONAL MEMOIRS OF U. S. GRANT, complete       
   Snippet: Although hailing from Illinois myself, the State of the President, I 
never met Mr. Lincoln until called to the capital to receive my commission as lieutenant-general.  I knew him, however, very well and favorably from the accounts 
given by officers u...

[p_284714] Score: 0.5579 | Book: THE SLEEPER AWAKES
   Snippet: "You have the world to choose from," said Lincoln; "whatever you want is yours."

[p_399388] Score: 0.5579 | Book: WHEN THE SLEEPER WAKES
   Snippet: "You have the world to choose from," said Lincoln; "whatever you want is yours."

[p_749197] Score: 0.5500 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN      
   Snippet: THE WRITINGS OF A. LINCOLN, Volume Seven, 1863-1865

[p_276527] Score: 0.5431 | Book: WOODSTOCK; OR, THE CAVALIER
   Snippet: "The same--Gentleman; of Squattlesea-mere, in the moist county of Lincoln."

[p_629147] Score: 0.5413 | Book: BIOGRAPHIES OF WORKING MEN
   Snippet: In 1861, the great storm burst over the States.  In the preceding November, Abraham Lincoln had been elected President.  Lincoln was himself, like Garfield, a self-made man, who had risen from the very same pioneer labourer class;--a wood-cutter and ...

Enter search query: government

--- Searching for: 'government' ---
Search completed in 3.6969 seconds.

--- Top 10 Search Results ---

[p_945523] Score: 0.4510 | Book: SECOND TREATISE OF GOVERNMENT by JOHN LOCKE     
   Snippet: AN ESSAY CONCERNING THE TRUE ORIGINAL, EXTENT AND END OF CIVIL GOVERNMENT

[p_868682] Score: 0.4425 | Book: Mr. WELLS has also written the following novels:   Snippet: CLASS II. It is supposed that the common man _cannot_ govern, and that government therefore must be through the agency of Able Persons who may be classified under one of the following sub-heads, either as

[p_256574] Score: 0.4403 | Book: THE MONIKINS
   Snippet: "I find all this very extraordinary, your government being professedly a government of the mass!"

[p_231328] Score: 0.4151 | Book: BOHN'S STANDARD LIBRARY
   Snippet: _Addison_. The greatest theorists ... among those very people [the Greeks and Romans,] have given the preference to such a form of government, as that which obtains in this kingdom.--_Swift_. Yet, this we see is liable to be wholly corrupted.

[p_494799] Score: 0.3863 | Book: THE FRENCH REVOLUTION
   Snippet: GOVERNMENT, Maurepas's, bad state of French, French revolutionary, Danton on.

[p_229387] Score: 0.3674 | Book: THE
   Snippet: "God help the nation where self-government, in its literal sense, exists, Hugh! The term is conventional, and, properly viewed, means a government in 
which the source of authority is the body of the nation, and does not come from any other sovereign....

[p_983673] Score: 0.3669 | Book: THE
   Snippet: _Cur._ I'll make it out: Rebellion is an insurrection against the government; but they that have the power are actually the government; therefore, if 
the people have the power, the rebellion is in the king.

[p_446460] Score: 0.3628 | Book: THE TEACHER:
   Snippet: Or let us imagine the following scene to have been the commencement of the introduction of the principle of limited self-government, into a school.   

[p_749336] Score: 0.3541 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN      
   Snippet: I commend the benevolent institutions established or patronized by the Government in this District to your generous and fostering care.

[p_761649] Score: 0.3397 | Book: THE PAPERS AND WRITINGS OF ABRAHAM LINCOLN      
   Snippet: There was a collateral object in the introduction of that Nebraska policy, which was to clothe the people of the Territories with a superior degree of self-government, beyond what they had ever had before. The first object and the 
main one of conferr...

Enter search query:
```
</details>

---
## 💡 Key Skills & Learnings
This project was an opportunity to move beyond basic NLP tutorials and engage with the practical challenges of building an actual search system.

* **Architectural Design:** I designed and implemented a sophisticated **Retriever-Reranker pipeline**, demonstrating an understanding of how to balance trade-offs between speed (TF-IDF) and semantic accuracy (Transformers).
* **End-to-End NLP Workflow:** I handled the entire NLP pipeline: sourcing raw data, extensive text preprocessing, training a custom BPE tokenizer, building multiple complex indexes (sparse TF-IDF and dense FAISS), and developing the final application logic.
* **Performance Optimization:** I identified and fixed performance bottlenecks. The initial 15-second search time was reduced to sub-second speeds by implementing a **FAISS** index. The multi-hour model indexing time was reduced to minutes by leveraging **GPU acceleration in a cloud environment (Google Colab)**.
* **Problem Analysis:** I diagnosed and explained complex model behaviors, such as the semantic model's failure on OOV terms (`cyberpunk`), its weird but predictable results for gibberish queries and for empty queries.

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
Once the environment is set up, run the code which creates the indexes. You may run this in google colab for quicker results. (Estimated execution time on free colab tier: 1hour)

```bash
python create_indexes.py
```
When all index files are in place, run the main search engine application:

```bash
python search_engine.py
```
The program will load the indexes and present an interactive prompt for search queries.

---
## 📈 Future Improvements
* **Build a Web Interface:** Wrap the search engine in a simple Streamlit or Flask web application for a more user-friendly demo.
* **Quantitative Evaluation:** Develop a ground-truth dataset to formally evaluate the retriever and reranker using metrics like Precision@k and MRR.
* **Deployment:** Deploy the final Streamlit application to a free hosting service like Hugging Face Spaces to create a shareable live demo.
