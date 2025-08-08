## Word Embeddings

This project explores different methods for generating word embeddings.

### Setup
1. Ensure Python 3.9 or higher is installed and the python3 alias points to such a version.
2. Set up a virtualenv.
```bash
pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
```
3. And install the required packages.
```
pip install -r requirements.txt
```
### Structure

- `data/`: all files generated or used for data generation.
    - `fairy_tales/`: *(NOT COMMITTED)* binary .txt files for corpus
    - `indiv_word_representations/`: *(NOT COMMITTED)* generated using various representation methods, the vector for each instances of KQMW in the corpus. Generated in `indiv_word_representations.ipynb`.
    - `iterative_vectors/`: *(NOT COMMITTED)* stores vector representations across iterations, generated via `iterative_vectors.py`.
    - `kqmw_iterations/`: *(NOT COMMITTED)* measures how representations for KQMW generated using the iterative vectors method changes across many iterations. Generated in `iteration_data.ipynb`.
    - `fairytales_doc_tf-idf.json`: document-based TF-IDFs. Generated in `generate_tf-idfs_docuemnts.ipynb`. Not used.
    - `fairytales_tokenized.json`: tokenized + lemmatized corpus. Generated in `generate_tf-idfs_words.ipynb`.
    - `fairytales_word_bloom-filters.json`: *(NOT COMMITTED)* Bloom filter for words in corpus, generated in `generate_bloom_filters.ipynb`.
    - `kqmw_iteration.json`: *(NOT COMMITTED)* Generated in `iteration_data.ipynb` to consolidate representations of KQMW across iterations for use in PCA plots.
    - `n_neighbors.json`: *(NOT COMMITTED)* Helper file for use in `generate_tf-idfs_words.ipynb`.
    - `neighbor_frequencies.json`: *(NOT COMMITTED)* Helper file for use in `generate_tf-idfs_words.ipynb`.
    - `sentence_examples/`: examples of tokenized lemmatized sentences containing KQMW.
- `eval/`: files from https://github.com/stanfordnlp/GloVe for analogies tests.
- `pca/`: directory for saving generated PCA plots. 
- `generate_bloom_filters.ipynb`: generates bloom filters for each word in the corpus.
- `generate_tf-idfs_documents.ipynb`: generates tf-idfs using the documents method, not used.
- `generate_tf-idfs_words.ipynb`: generates tf-idfs using the words method.
- `indiv_word_representations.ipynb`: given final representations for words in corpus, generates instance representations for specific words for further analysis.
- `iteration_data.ipynb`: generates data for analyzing representations across iterations generated via `iterative_vectors.py`.
- `pca_plots.ipynb`: generates PCA plots.
- `iterative_vectors.py`: generates vector representations using the iterative method.