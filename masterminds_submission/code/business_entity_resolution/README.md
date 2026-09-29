# 🚀 Amazon ML Challenge: Entity Resolution & Record Linkage Pipeline

An optimized, end-to-end Machine Learning pipeline developed for the **Amazon ML Challenge** by **Umanshi Gupta** and **Shivika Bhawsar**. 

This project addresses large-scale entity resolution and record linkage across massive, noisy multi-source enterprise datasets under strict memory and computational constraints.

---

## 📌 Project Overview
Entity resolution (ER) across large-scale heterogeneous databases frequently leads to combinatorial explosion and severe memory bottlenecks during feature engineering. This project implements a high-performance blocking and classification architecture designed to match and deduplicate entity records accurately and efficiently.

### **Key Achievements & Highlights:**
* **Massive-Scale Ingestion:** Successfully engineered a data pipeline capable of parsing, cleaning, and normalizing multi-source entity data across **12M+ rows**.
* **Memory Optimization:** Overcame out-of-memory (RAM) overflow issues by replacing heavy, memory-intensive TF-IDF matrix generation with vectorized string-similarity metrics (`RapidFuzz`).
* **Machine Learning Classification:** Trained a robust scikit-learn Random Forest classifier with optimized blocking thresholds, achieving a validated prediction threshold of **0.85**.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **String Similarity:** RapidFuzz (Jaro-Winkler, Levenshtein Distance, Token Jaccard Similarity)
* **Model Serialization:** Joblib

---

## ⚙️ Architecture & Pipeline
1. **Preprocessing (`preprocess.py`):** Cleans and standardizes entity names, addresses, and country attributes across disparate sources.
2. **Blocking (`blocking.py`):** Employs token and prefix blocking keys (`key_2tokens`, `key_prefix3`) while automatically purging overly generic, high-frequency tokens to restrict the search space and boost recall.
3. **Pairwise Feature Engineering (`features.py`):** Calculates pairwise string distances and similarity scores for candidate entity pairs.
4. **Classification & Prediction (`train_model.py` & `predict.py`):** Utilizes the trained Random Forest model with optimized thresholds (`0.85`) to generate production-ready submission files (`candidate_pairs.tsv` and `matching_results.tsv`).

