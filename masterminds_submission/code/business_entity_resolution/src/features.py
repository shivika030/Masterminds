import time
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz

def _token_jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def build_pairwise_features(s1_df: pd.DataFrame, other_df: pd.DataFrame, candidate_pairs_df: pd.DataFrame) -> pd.DataFrame:
    t_start = time.time()
    
    exploded = candidate_pairs_df.assign(
        candidate_entity_id=candidate_pairs_df["candidate_entity_ids"].str.split(",")
    ).explode("candidate_entity_id")
    exploded = exploded[exploded["candidate_entity_id"].notna() & (exploded["candidate_entity_id"] != "")]
    exploded = exploded.drop(columns=["candidate_entity_ids"]).reset_index(drop=True)
    
    if len(exploded) == 0:
        cols = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    s1_cols = s1_df[["entity_id", "clean_name", "clean_address", "country"]].rename(
        columns={c: f"s1_{c}" for c in ["clean_name", "clean_address", "country"]}
    ).rename(columns={"entity_id": "source1_entity_id"})
    
    other_cols = other_df[["entity_id", "clean_name", "clean_address", "country"]].rename(
        columns={c: f"cand_{c}" for c in ["clean_name", "clean_address", "country"]}
    ).rename(columns={"entity_id": "candidate_entity_id"})
    
    pairs = exploded.merge(s1_cols, on="source1_entity_id", how="left")
    pairs = pairs.merge(other_cols, on="candidate_entity_id", how="left")
    pairs = pairs.dropna(subset=["cand_clean_name"]).reset_index(drop=True)

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

    return pairs

FEATURE_COLUMNS = [
    "name_jaccard",
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "name_length_diff",
    "address_jaccard",
    "address_levenshtein_ratio",
    "country_match",
]

def attach_labels(pairs_features_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
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