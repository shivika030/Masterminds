from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    CANDIDATE_PAIRS_PATH, OUTPUT_DIR,
    MAX_CANDIDATES_PER_ENTITY, TFIDF_NGRAM_RANGE, TFIDF_TOP_K,
)
from normalizer import normalize_name, normalize_address


# ---------------------------------------------------------------------------
# 1. Load + normalize one source file into a standard frame
# ---------------------------------------------------------------------------
def load_and_normalize(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    name_info = df["business_name"].apply(normalize_name)
    df["clean_name"] = name_info.apply(lambda x: x["clean_name"])
    df["full_normalized_name"] = name_info.apply(lambda x: x["full_normalized"])
    df["clean_address"] = df["business_address"].apply(normalize_address)
    df["blocking_key"] = (
        df["clean_name"].str.split().str[:2].str.join(" ")  # first 2 tokens
        + "|" + df["country"].str.lower().str.strip()
    )
    return df


# ---------------------------------------------------------------------------
# 2. Strategy A — exact/loose token-key blocking (dict lookup, very fast)
# ---------------------------------------------------------------------------
def token_key_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame) -> dict:
    """Returns {s1_entity_id: set(other_entity_ids)} for entities sharing a blocking_key."""
    key_to_ids = defaultdict(set)
    for _, row in other_df.iterrows():
        key_to_ids[row["blocking_key"]].add(row["entity_id"])

    result = {}
    for _, row in s1_df.iterrows():
        result[row["entity_id"]] = set(key_to_ids.get(row["blocking_key"], set()))
    return result


# ---------------------------------------------------------------------------
# 3. Strategy B — TF-IDF character n-gram cosine similarity (nearest neighbors)
# ---------------------------------------------------------------------------
def tfidf_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame, top_k=TFIDF_TOP_K) -> dict:
    """Returns {s1_entity_id: set(other_entity_ids)} using name+address text similarity."""
    if len(other_df) == 0:
        return {eid: set() for eid in s1_df["entity_id"]}

    s1_text = (s1_df["full_normalized_name"] + " " + s1_df["clean_address"]).tolist()
    other_text = (other_df["full_normalized_name"] + " " + other_df["clean_address"]).tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=TFIDF_NGRAM_RANGE, min_df=1)
    other_matrix = vectorizer.fit_transform(other_text)
    s1_matrix = vectorizer.transform(s1_text)

    k = min(top_k, len(other_df))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(other_matrix)
    distances, indices = nn.kneighbors(s1_matrix)

    other_ids = other_df["entity_id"].values
    result = {}
    for i, s1_id in enumerate(s1_df["entity_id"].values):
        result[s1_id] = set(other_ids[idx] for idx in indices[i])
    return result


# ---------------------------------------------------------------------------
# 4. Combine strategies (union) and cap per entity
# ---------------------------------------------------------------------------
def union_and_cap(*candidate_dicts, cap=MAX_CANDIDATES_PER_ENTITY) -> dict:
    combined = defaultdict(set)
    for d in candidate_dicts:
        for eid, matches in d.items():
            combined[eid] |= matches
    # NOTE: capping here is naive (arbitrary truncation). Once you have a
    # similarity score, sort by score descending before capping instead.
    return {eid: set(list(matches)[:cap]) for eid, matches in combined.items()}


# ---------------------------------------------------------------------------
# 5. Main — build candidates against BOTH Source 2 and Source 3, write TSV
# ---------------------------------------------------------------------------
def build_candidate_pairs(s1_df, s2_df, s3_df) -> pd.DataFrame:
    s2_token = token_key_candidates(s1_df, s2_df)
    s3_token = token_key_candidates(s1_df, s3_df)
    s2_tfidf = tfidf_candidates(s1_df, s2_df)
    s3_tfidf = tfidf_candidates(s1_df, s3_df)

    merged = union_and_cap(s2_token, s3_token, s2_tfidf, s3_tfidf)

    rows = []
    for eid in s1_df["entity_id"]:
        candidates = merged.get(eid, set())
        rows.append({
            "source1_entity_id": eid,
            "candidate_entity_ids": ",".join(sorted(candidates)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading and normalizing test sources...")
    s1 = load_and_normalize(TEST_SOURCE1)
    s2 = load_and_normalize(TEST_SOURCE2)
    s3 = load_and_normalize(TEST_SOURCE3)

    print(f"Source1: {len(s1)} | Source2: {len(s2)} | Source3: {len(s3)}")
    print("Building candidate pairs (this may take a bit for TF-IDF)...")

    result_df = build_candidate_pairs(s1, s2, s3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(CANDIDATE_PAIRS_PATH, sep="\t", index=False)

    empty_count = (result_df["candidate_entity_ids"] == "").sum()
    avg_candidates = result_df["candidate_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    ).mean()
    print(f"Wrote {len(result_df)} rows to {CANDIDATE_PAIRS_PATH}")
    print(f"Entities with zero candidates: {empty_count}")
    print(f"Average candidates per entity: {avg_candidates:.1f}")