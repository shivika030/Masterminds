# Business Entity Resolution - Methodology Documentation

## 1. Methodology Used
Our approach to the Entity Resolution challenge utilizes a purely unsupervised, NLP-based statistical matching pipeline. Given the noisy nature of the business names and addresses (typos, abbreviations, missing components), we engineered a solution based on character-level Term Frequency-Inverse Document Frequency (TF-IDF) combined with Cosine Similarity. 

By concatenating the `business_name` and `business_address` into a single text feature, the model evaluates the holistic similarity of entities across Source 1, Source 2, and Source 3, effectively bypassing the need for external databases or explicit geographical parsing.

## 2. Candidate Generation / Blocking Strategy
To generate the `candidate_pairs.tsv` and reduce the $O(N^2)$ comparison space, we implemented a K-Nearest Neighbors blocking strategy using our TF-IDF vector space:
* **Vectorization Context:** We fit the TF-IDF vectorizer on the combined text of all records in Source 2 and Source 3 to create a comprehensive vocabulary of noise patterns and n-grams present in the target sources.
* **Retrieval:** For every entity in Source 1, we computed its cosine similarity against the entire Source 2/Source 3 matrix.
* **Candidate Pool:** We selected the Top $K$ ($K=15$) highest-scoring entities from Source 2/Source 3 as our candidate blocks. This guarantees a high recall ceiling while significantly shrinking the search space for the final matching threshold.

## 3. Model Architecture and Feature Engineering
* **Feature Engineering:** 
  * Missing values (NaNs) in names or addresses were imputed with empty strings.
  * All text was converted to lowercase to handle capitalization inconsistencies.
  * We extracted **Character N-grams** (sizes 2 to 4) using `analyzer='char_wb'` (character n-grams inside word boundaries). This is highly resilient to transliteration variants, typos, and municipal numbering formats compared to standard word-level tokens.
* **Model Architecture:** 
  * We utilized Scikit-Learn's `TfidfVectorizer` and `cosine_similarity`. 
  * **Matching Threshold:** From the 15 candidates generated during blocking, the final matching model applies a strict Cosine Similarity threshold of `0.75`. Candidates scoring $\ge 0.75$ are classified as matches, while others are discarded.
  * **Parameters:** This model uses 0 trainable parameters, relying entirely on statistical corpus frequencies, strictly adhering to the "under 8 Billion parameters" and MIT/Apache 2.0 license constraints.

## 4. Additional Relevant Information
* **Memory Optimization:** To prevent Out-Of-Memory (OOM) errors when processing the dense similarity matrix for the entire test set, the inference pipeline batches Source 1 queries (batch size = 1000).
* **Singleton Handling:** Our thresholding logic ensures that if no candidate in the Top 15 meets the 0.75 similarity score, the entity is correctly treated as a singleton (returning an empty string), which optimizes for the precision-heavy $F_{0.5}$ metric.