import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _token_jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _tfidf_cosine_pairs(texts_a: list, texts_b: list) -> np.ndarray:
    """
    Row-wise cosine similarity between texts_a[i] and texts_b[i], WITHOUT
    building a full pairwise matrix (memory-safe for large candidate sets).
    Fits one shared vectorizer on the combined corpus so both sides share
    a vocabulary.
    """
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    vectorizer.fit(texts_a + texts_b)
    mat_a = vectorizer.transform(texts_a)
    mat_b = vectorizer.transform(texts_b)

    # row-wise dot product / norms = row-wise cosine similarity
    dot = np.array(mat_a.multiply(mat_b).sum(axis=1)).flatten()
    norm_a = np.sqrt(np.array(mat_a.multiply(mat_a).sum(axis=1)).flatten())
    norm_b = np.sqrt(np.array(mat_b.multiply(mat_b).sum(axis=1)).flatten())
    denom = norm_a * norm_b
    denom[denom == 0] = 1e-9  # avoid divide-by-zero for empty strings
    return dot / denom


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_pairwise_features(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, candidate_pairs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    s1_df, other_df: normalized frames from blocking.load_and_normalize()
                      (other_df should be Source2+Source3 concatenated)
    candidate_pairs_df: output of blocking.py, columns
                         [source1_entity_id, candidate_entity_ids]
    Returns one row per (source1_entity_id, candidate_entity_id) pair with
    feature columns, ready for model training/inference.
    """
    # explode "a,b,c" candidate lists into one row per pair
    exploded = candidate_pairs_df.assign(
        candidate_entity_id=candidate_pairs_df["candidate_entity_ids"].str.split(",")
    ).explode("candidate_entity_id")
    exploded = exploded[exploded["candidate_entity_id"].notna() & (exploded["candidate_entity_id"] != "")]
    exploded = exploded.drop(columns=["candidate_entity_ids"]).reset_index(drop=True)

    s1_cols = s1_df[["entity_id", "clean_name", "full_normalized_name", "clean_address", "country"]].rename(
        columns={c: f"s1_{c}" for c in ["clean_name", "full_normalized_name", "clean_address", "country"]}
    ).rename(columns={"entity_id": "source1_entity_id"})

    other_cols = other_df[["entity_id", "clean_name", "full_normalized_name", "clean_address", "country"]].rename(
        columns={c: f"cand_{c}" for c in ["clean_name", "full_normalized_name", "clean_address", "country"]}
    ).rename(columns={"entity_id": "candidate_entity_id"})

    pairs = exploded.merge(s1_cols, on="source1_entity_id", how="left")
    pairs = pairs.merge(other_cols, on="candidate_entity_id", how="left")

    # drop any pairs where the candidate id didn't exist in other_df (safety)
    pairs = pairs.dropna(subset=["cand_clean_name"]).reset_index(drop=True)

    if len(pairs) == 0:
        return pairs.assign(**{col: [] for col in [
            "name_jaccard", "name_levenshtein_ratio", "name_jaro_winkler",
            "name_tfidf_cosine", "address_jaccard", "address_levenshtein_ratio",
            "address_tfidf_cosine", "country_match", "name_length_diff",
        ]})

    # --- name features ---
    pairs["name_jaccard"] = pairs.apply(
        lambda r: _token_jaccard(r["s1_clean_name"], r["cand_clean_name"]), axis=1
    )
    pairs["name_levenshtein_ratio"] = pairs.apply(
        lambda r: fuzz.ratio(r["s1_clean_name"], r["cand_clean_name"]) / 100.0, axis=1
    )
    pairs["name_jaro_winkler"] = pairs.apply(
        lambda r: JaroWinkler.normalized_similarity(r["s1_clean_name"], r["cand_clean_name"]), axis=1
    )
    pairs["name_tfidf_cosine"] = _tfidf_cosine_pairs(
        pairs["s1_full_normalized_name"].tolist(), pairs["cand_full_normalized_name"].tolist()
    )
    pairs["name_length_diff"] = (
        pairs["s1_clean_name"].str.len() - pairs["cand_clean_name"].str.len()
    ).abs()

    # --- address features ---
    pairs["address_jaccard"] = pairs.apply(
        lambda r: _token_jaccard(r["s1_clean_address"], r["cand_clean_address"]), axis=1
    )
    pairs["address_levenshtein_ratio"] = pairs.apply(
        lambda r: fuzz.ratio(r["s1_clean_address"], r["cand_clean_address"]) / 100.0, axis=1
    )
    pairs["address_tfidf_cosine"] = _tfidf_cosine_pairs(
        pairs["s1_clean_address"].tolist(), pairs["cand_clean_address"].tolist()
    )

    # --- country feature ---
    pairs["country_match"] = (
        pairs["s1_country"].str.lower().str.strip() == pairs["cand_country"].str.lower().str.strip()
    ).astype(int)

    return pairs


FEATURE_COLUMNS = [
    "name_jaccard", "name_levenshtein_ratio", "name_jaro_winkler", "name_tfidf_cosine",
    "name_length_diff", "address_jaccard", "address_levenshtein_ratio",
    "address_tfidf_cosine", "country_match",
]


# ---------------------------------------------------------------------------
# Label attachment (training only — test has no ground truth)
# ---------------------------------------------------------------------------
def attach_labels(pairs_features_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    ground_truth_df: columns [source1_entity_id, matched_entity_ids] (comma-separated)
    Adds a binary "label" column: 1 if (source1_entity_id, candidate_entity_id)
    is a true match, else 0.
    """
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        matched = str(row["matched_entity_ids"]).strip()
        gt_map[row["source1_entity_id"]] = set(matched.split(",")) if matched else set()

    def is_match(row):
        true_matches = gt_map.get(row["source1_entity_id"], set())
        return int(row["candidate_entity_id"] in true_matches)

    pairs_features_df = pairs_features_df.copy()
    pairs_features_df["label"] = pairs_features_df.apply(is_match, axis=1)
    return pairs_features_df


if __name__ == "__main__":
    # tiny smoke test with synthetic data
    s1 = pd.DataFrame([
        {"entity_id": "S1-001", "clean_name": "acme corp", "full_normalized_name": "acme corporation",
         "clean_address": "123 main road", "country": "US"},
    ])
    other = pd.DataFrame([
        {"entity_id": "S2-001", "clean_name": "acme", "full_normalized_name": "acme incorporated",
         "clean_address": "123 main road", "country": "US"},
        {"entity_id": "S3-001", "clean_name": "globex", "full_normalized_name": "globex corporation",
         "clean_address": "999 other street", "country": "IN"},
    ])
    candidates = pd.DataFrame([{"source1_entity_id": "S1-001", "candidate_entity_ids": "S2-001,S3-001"}])

    feats = build_pairwise_features(s1, other, candidates)
    print(feats[["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS])