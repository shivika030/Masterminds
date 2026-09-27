import time

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_BATCH_SIZE = 200_000  # pairs processed per batch during cosine lookup


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


# ---------------------------------------------------------------------------
# Entity-level TF-IDF embedding — fit + transform ONCE per entity, reused
# across every pair that entity appears in.
# ---------------------------------------------------------------------------
def _build_entity_embeddings(s1_df: pd.DataFrame, other_df: pd.DataFrame, text_col: str):
    """
    Fits one TF-IDF vectorizer on all entity texts (s1 + other combined),
    transforms each side once, and returns:
        matrix, id_to_row: dict mapping entity_id -> row index in `matrix`
    matrix's first len(s1_df) rows correspond to s1_df (in order), and the
    remaining rows correspond to other_df (in order).
    """
    s1_texts = s1_df[text_col].fillna("").tolist()
    other_texts = other_df[text_col].fillna("").tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(s1_texts + other_texts)

    id_to_row = {}
    for i, eid in enumerate(s1_df["entity_id"].tolist()):
        id_to_row[eid] = i
    offset = len(s1_df)
    for i, eid in enumerate(other_df["entity_id"].tolist()):
        id_to_row[eid] = offset + i

    return matrix, id_to_row


def _batched_cosine(matrix, idx_a: np.ndarray, idx_b: np.ndarray, batch_size=TFIDF_BATCH_SIZE) -> np.ndarray:
    """Row-wise cosine similarity for paired row indices, processed in memory-safe batches."""
    n = len(idx_a)
    result = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a_batch = matrix[idx_a[start:end]]
        b_batch = matrix[idx_b[start:end]]
        dot = np.array(a_batch.multiply(b_batch).sum(axis=1)).flatten()
        norm_a = np.sqrt(np.array(a_batch.multiply(a_batch).sum(axis=1)).flatten())
        norm_b = np.sqrt(np.array(b_batch.multiply(b_batch).sum(axis=1)).flatten())
        denom = norm_a * norm_b
        denom[denom == 0] = 1e-9
        result[start:end] = dot / denom
    return result


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_pairwise_features(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, candidate_pairs_df: pd.DataFrame
) -> pd.DataFrame:
    t_start = time.time()
    print("  Exploding candidate pairs...", flush=True)

    exploded = candidate_pairs_df.assign(
        candidate_entity_id=candidate_pairs_df["candidate_entity_ids"].str.split(",")
    ).explode("candidate_entity_id")
    exploded = exploded[exploded["candidate_entity_id"].notna() & (exploded["candidate_entity_id"] != "")]
    exploded = exploded.drop(columns=["candidate_entity_ids"]).reset_index(drop=True)
    print(f"    {len(exploded)} pairs total", flush=True)

    if len(exploded) == 0:
        cols = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    print("  Merging entity attributes onto pairs...", flush=True)
    s1_cols = s1_df[["entity_id", "clean_name", "full_normalized_name", "clean_address", "country"]].rename(
        columns={c: f"s1_{c}" for c in ["clean_name", "full_normalized_name", "clean_address", "country"]}
    ).rename(columns={"entity_id": "source1_entity_id"})

    other_cols = other_df[["entity_id", "clean_name", "full_normalized_name", "clean_address", "country"]].rename(
        columns={c: f"cand_{c}" for c in ["clean_name", "full_normalized_name", "clean_address", "country"]}
    ).rename(columns={"entity_id": "candidate_entity_id"})

    pairs = exploded.merge(s1_cols, on="source1_entity_id", how="left")
    pairs = pairs.merge(other_cols, on="candidate_entity_id", how="left")
    pairs = pairs.dropna(subset=["cand_clean_name"]).reset_index(drop=True)
    print(f"    {len(pairs)} pairs after merge ({time.time()-t_start:.1f}s elapsed)", flush=True)

    # -----------------------------------------------------------------
    # String-similarity metrics — zip-based loop (fast, NOT DataFrame.apply)
    # -----------------------------------------------------------------
    print("  Computing string-similarity metrics (zip loop)...", flush=True)
    t0 = time.time()
    s1_names = pairs["s1_clean_name"].tolist()
    cand_names = pairs["cand_clean_name"].tolist()
    s1_addrs = pairs["s1_clean_address"].tolist()
    cand_addrs = pairs["cand_clean_address"].tolist()
    s1_countries = pairs["s1_country"].str.lower().str.strip().tolist()
    cand_countries = pairs["cand_country"].str.lower().str.strip().tolist()

    name_jaccard, name_lev, name_jw, name_len_diff = [], [], [], []
    addr_jaccard, addr_lev, country_match = [], [], []

    for sn, cn, sa, ca, sc, cc in zip(s1_names, cand_names, s1_addrs, cand_addrs, s1_countries, cand_countries):
        name_jaccard.append(_token_jaccard(sn, cn))
        name_lev.append(fuzz.ratio(sn, cn) / 100.0)
        name_jw.append(JaroWinkler.normalized_similarity(sn, cn))
        name_len_diff.append(abs(len(sn) - len(cn)))
        addr_jaccard.append(_token_jaccard(sa, ca))
        addr_lev.append(fuzz.ratio(sa, ca) / 100.0)
        country_match.append(1 if sc == cc else 0)

    pairs["name_jaccard"] = name_jaccard
    pairs["name_levenshtein_ratio"] = name_lev
    pairs["name_jaro_winkler"] = name_jw
    pairs["name_length_diff"] = name_len_diff
    pairs["address_jaccard"] = addr_jaccard
    pairs["address_levenshtein_ratio"] = addr_lev
    pairs["country_match"] = country_match
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)

    # -----------------------------------------------------------------
    # TF-IDF cosine — entity-level embeddings, batched pair lookup
    # -----------------------------------------------------------------
    print("  Building entity-level TF-IDF embeddings (name)...", flush=True)
    t0 = time.time()
    name_matrix, name_id_to_row = _build_entity_embeddings(s1_df, other_df, "full_normalized_name")
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)

    print("  Building entity-level TF-IDF embeddings (address)...", flush=True)
    t0 = time.time()
    addr_matrix, addr_id_to_row = _build_entity_embeddings(s1_df, other_df, "clean_address")
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)

    print("  Computing TF-IDF cosine similarity (batched)...", flush=True)
    t0 = time.time()
    s1_ids = pairs["source1_entity_id"].tolist()
    cand_ids = pairs["candidate_entity_id"].tolist()

    name_idx_a = np.array([name_id_to_row[e] for e in s1_ids])
    name_idx_b = np.array([name_id_to_row[e] for e in cand_ids])
    pairs["name_tfidf_cosine"] = _batched_cosine(name_matrix, name_idx_a, name_idx_b)

    addr_idx_a = np.array([addr_id_to_row[e] for e in s1_ids])
    addr_idx_b = np.array([addr_id_to_row[e] for e in cand_ids])
    pairs["address_tfidf_cosine"] = _batched_cosine(addr_matrix, addr_idx_a, addr_idx_b)
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Total feature-building time: {time.time()-t_start:.1f}s", flush=True)
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
    is a true match, else 0. Vectorized via merge, not per-row apply.
    """
    gt_exploded = ground_truth_df.assign(
        candidate_entity_id=ground_truth_df["matched_entity_ids"].str.split(",")
    ).explode("candidate_entity_id")
    gt_exploded = gt_exploded[gt_exploded["candidate_entity_id"].notna() & (gt_exploded["candidate_entity_id"] != "")]
    gt_exploded = gt_exploded[["source1_entity_id", "candidate_entity_id"]].copy()
    gt_exploded["label"] = 1

    merged = pairs_features_df.merge(
        gt_exploded, on=["source1_entity_id", "candidate_entity_id"], how="left"
    )
    merged["label"] = merged["label"].fillna(0).astype(int)
    return merged


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