import time
import gc
from collections import defaultdict
import pandas as pd

from config import (
    TEST_SOURCE1, TEST_SOURCE2,
    CANDIDATE_PAIRS_PATH, OUTPUT_DIR,
    MAX_CANDIDATES_PER_ENTITY, MAX_BLOCK_SIZE, S1_CHUNK_SIZE, MAX_CHUNK_MERGE_ROWS,
)
# Removed normalize_address_series as it consumes massive RAM and isn't used for blocking
from normalizer import normalize_name_series

def load_and_normalize(path) -> pd.DataFrame:
    t0 = time.time()
    print(f"  reading {path} ...", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    print(f"    {len(df)} rows, normalizing names...", flush=True)

    name_info = normalize_name_series(df["business_name"])
    df["clean_name"] = name_info["clean_name"]
    country_norm = df["country"].str.lower().str.strip()

    df["key_2tokens"] = (
        df["clean_name"].str.split().str[:2].str.join(" ") + "|" + country_norm
    )
    df["key_prefix5"] = (
        df["clean_name"].str.replace(" ", "", regex=False).str[:5] + "|" + country_norm
    )

    # CRITICAL MEMORY FIX: Drop heavy text columns immediately
    df = df[["entity_id", "key_2tokens", "key_prefix5"]]

    print(f"    done: {path}  ({time.time()-t0:.1f}s)", flush=True)
    return df


def merge_block(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, key_col: str,
    max_block_size: int = MAX_BLOCK_SIZE, max_s1_chunk_size: int = S1_CHUNK_SIZE,
    target_chunk_merge_rows: int = MAX_CHUNK_MERGE_ROWS,
) -> dict:
    key_counts = other_df[key_col].value_counts()
    oversized_keys = key_counts[key_counts > max_block_size]
    if len(oversized_keys) > 0:
        print(f"    purging {len(oversized_keys)} oversized key(s)", flush=True)
        
    surviving_counts = key_counts[key_counts <= max_block_size]
    valid_keys = set(surviving_counts.index)

    other_small = other_df[other_df[key_col].isin(valid_keys)][["entity_id", key_col]].rename(
        columns={"entity_id": "candidate_id"}
    )
    if len(other_small) == 0:
        return {}

    max_remaining_block = int(surviving_counts.max()) if len(surviving_counts) > 0 else 1
    adaptive_chunk_size = max(1, min(max_s1_chunk_size, target_chunk_merge_rows // max_remaining_block))

    result = {}
    n = len(s1_df)
    for start in range(0, n, adaptive_chunk_size):
        end = min(start + adaptive_chunk_size, n)
        s1_chunk = s1_df.iloc[start:end][["entity_id", key_col]]
        merged = s1_chunk.merge(other_small, on=key_col, how="inner")
        if len(merged) == 0:
            continue
        grouped = merged.groupby("entity_id")["candidate_id"].apply(set)
        result.update(grouped.to_dict())
    return result


def union_and_cap(*candidate_dicts, cap=MAX_CANDIDATES_PER_ENTITY) -> dict:
    agreement_counts = defaultdict(lambda: defaultdict(int))
    for d in candidate_dicts:
        for eid, matches in d.items():
            for cid in matches:
                agreement_counts[eid][cid] += 1

    result = {}
    for eid, counts in agreement_counts.items():
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        result[eid] = set(cid for cid, _ in ranked[:cap])
    return result


def build_candidate_pairs(s1_df, s2_df) -> pd.DataFrame:
    print("Blocking against Source 2 (key_2tokens)...", flush=True)
    t0 = time.time()
    s2_key2 = merge_block(s1_df, s2_df, "key_2tokens")
    print(f"  -> pairs found ({time.time()-t0:.1f}s)", flush=True)

    print("Blocking against Source 2 (key_prefix5)...", flush=True)
    t0 = time.time()
    s2_prefix = merge_block(s1_df, s2_df, "key_prefix5")
    print(f"  -> pairs found ({time.time()-t0:.1f}s)", flush=True)

    print("Combining and capping candidates per entity...", flush=True)
    merged = union_and_cap(s2_key2, s2_prefix)

    rows = []
    for eid in s1_df["entity_id"]:
        candidates = merged.get(eid, set())
        rows.append({
            "source1_entity_id": eid,
            "candidate_entity_ids": ",".join(sorted(candidates)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    overall_start = time.time()
    print("Loading and normalizing Source 1...", flush=True)
    s1 = load_and_normalize(TEST_SOURCE1)
    
    print("Loading and normalizing Source 2...", flush=True)
    s2 = load_and_normalize(TEST_SOURCE2)

    print(f"Source1: {len(s1)} | Source2: {len(s2)}", flush=True)

    result_df = build_candidate_pairs(s1, s2)
    
    # Free memory before saving output
    del s1
    del s2
    gc.collect()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(CANDIDATE_PAIRS_PATH, sep="\t", index=False)

    empty_count = (result_df["candidate_entity_ids"] == "").sum()
    avg_candidates = result_df["candidate_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    ).mean()
    
    print(f"\nWrote {len(result_df)} rows to {CANDIDATE_PAIRS_PATH}")
    print(f"Entities with zero candidates: {empty_count} ({empty_count/len(result_df)*100:.1f}%)")
    print(f"Average candidates per entity: {avg_candidates:.1f}")
    print(f"Total time: {time.time()-overall_start:.1f}s")