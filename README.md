# 🚀 Amazon ML Challenge: Entity Resolution & Record Linkage Pipeline

An optimized, end-to-end Machine Learning pipeline developed for the **Amazon ML Challenge** by **Umanshi Gupta** and **Shivika Bhawsar**. 

This project addresses large-scale entity resolution and record linkage across massive, noisy multi-source enterprise datasets under strict memory and computational constraints.

---

## 📌 Project Overview
Entity resolution (ER) across large-scale heterogeneous databases frequently leads to combinatorial explosion and severe memory bottlenecks during feature engineering. This project implements a high-performance blocking and classification architecture designed to match and deduplicate entity records accurately and efficiently.

### **Key Achievements & Highlights:**
- **Massive-Scale Ingestion:** Successfully engineered a data pipeline capable of parsing, cleaning, and normalizing multi-source entity data across **12M+ rows**.
- **Memory Optimization:** Overcame out-of-memory (RAM) overflow issues by replacing heavy, memory-intensive TF-IDF matrix generation with vectorized string-similarity metrics (`RapidFuzz`).
- **Machine Learning Classification:** Trained a robust scikit-learn Random Forest classifier with optimized blocking thresholds, achieving a validated prediction threshold of **0.85**.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest Classifier)
- **String Similarity:** RapidFuzz (Jaro-Winkler, Levenshtein Distance, Token Jaccard Similarity)
- **Model Serialization:** Joblib

---

## 📊 Scalability & Memory Management
This pipeline is designed to perform entity resolution across datasets totaling over 12 million rows. Processing text data at this scale within a single-node cloud environment introduces strict hardware constraints, specifically Kaggle's 30GB RAM limit. 

Because Pandas executes entirely in-memory, attempting to load, normalize, and compute the blocking pairs for Source 1 (1.7M rows), Source 2 (4.8M rows), and Source 3 (5M+ rows) sequentially within a single session exceeds available memory, causing the system to swap to disk and crash.

To achieve a clean, end-to-end execution within this specific hardware boundary, the current Kaggle implementation scopes the processing to **Source 1 and Source 2 (processing ~6.5 million rows total)**. 

**In-Memory Optimizations Implemented:**
* **Aggressive Column Projection:** Dropping heavy text columns (addresses, unnormalized names) immediately after generating the `key_2tokens` and `key_prefix3` blocking keys to minimize DataFrame footprint.
* **Manual State Clearing:** Explicitly removing processed DataFrames (`del s2`) and manually triggering Python's garbage collector (`gc.collect()`) to reclaim RAM before allocating memory for the next phase.
* **Chunked Merging:** Executing the `merge_block` inner joins in defined batches to cap peak memory spikes during the Cartesian product generation.

---

## ⚙️ Architecture & Pipeline
1. **Preprocessing (`preprocess.py`):** Cleans and standardizes entity names, addresses, and country attributes across disparate sources.
2. **Blocking (`blocking.py`):** Employs token and prefix blocking keys (`key_2tokens`, `key_prefix3`) while automatically purging overly generic, high-frequency tokens to restrict the search space and boost recall.
3. **Pairwise Feature Engineering (`features.py`):** Calculates pairwise string distances and similarity scores for candidate entity pairs.
4. **Classification & Prediction (`train_model.py` & `predict.py`):** Utilizes the trained Random Forest model with optimized thresholds (`0.85`) to generate production-ready submission files (`candidate_pairs.tsv` and `matching_results.tsv`).