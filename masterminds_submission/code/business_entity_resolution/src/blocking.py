import time
import gc
from collections import defaultdict
import pandas as pd

from config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    CANDIDATE_PAIRS_PATH, OUTPUT_DIR,
    MAX_CANDIDATES_PER_ENTITY, MAX_BLOCK_SIZE, S1_CHUNK_SIZE,
)
# Note: I removed normalize_address_series since it consumes massive RAM 
# and is not used to create your blocking keys.
from normalizer import normalize_name_series

def load_and_normalize(path, nrows=None) -> pd.DataFrame:
    t0 = time.time()
    print(f"  reading {path} ...", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows).fillna("")
    print(f"    {len(df)} rows, normalizing names...", flush=True)
    
    name_info = normalize_name_series(df["business_name"])
    df["clean_name"] = name_info["clean_name"]
    
    country_norm = df["country"].str.lower().str.strip()

    df["key_2tokens"] = (
        df["clean_name"].str.split().str[:2].str.join(" ") + "|" + country_norm
    )
    df["key_prefix3"] = (
        df["clean_name"].str.replace(" ", "", regex=False).str[:3] + "|" + country_norm
    )
    
    # CRITICAL MEMORY FIX: Keep ONLY the columns needed for blocking.
    # This drops business_name, business_address, etc., freeing gigabytes of RAM.
    df = df[["entity_id", "key_2tokens", "key_prefix3"]]
    
    print(f"    done: {path}  ({time.time()-t0:.1f}s)", flush=True)
    return df

def merge_block(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    key_col: str,
    max_block_size: int = MAX_BLOCK_SIZE,
    s1_chunk_size: int = S1_CHUNK_SIZE,
) -> dict:
    key_counts = other_df[key_col].value_counts()
    oversized_keys = key_counts[key_counts > max_block_size]
    
    if len(oversized_keys) > 0:
        print(f"    purging {len(oversized_keys)} oversized key(s) "
              f"(covering {oversized_keys.sum()} rows, e.g. {oversized_keys.index[0]!r}: "
              f"{oversized_keys.iloc[0]} matches) — too generic to be useful blocking", flush=True)
              
    valid_keys = set(key_counts[key_counts <= max_block_size].index)
    other_small = other_df[other_df[key_col].isin(valid_keys)][["entity_id", key_col]].rename(
        columns={"entity_id": "candidate_id"}
    )
    
    if len(other_small) == 0:
        return {}

    result = {}
    n = len(s1_df)
    for start in range(0, n, s1_chunk_size):
        end = min(start + s1_chunk_size, n)
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


if __name__ == "__main__":
    overall_start = time.time()
    
    # Set to a number like 10000 to test without crashing. Set to None for the full run.
    TESTING_ROWS = None 
    
    print("Loading Source 1...", flush=True)
    s1 = load_and_normalize(TEST_SOURCE1, nrows=TESTING_ROWS)
    
    print("\n--- Blocking against Source 2 ---", flush=True)
    s2 = load_and_normalize(TEST_SOURCE2, nrows=TESTING_ROWS)
    
    t0 = time.time()
    s2_key2 = merge_block(s1, s2, "key_2tokens")
    print(f"  -> key_2tokens pairs found ({time.time()-t0:.1f}s)", flush=True)
    
    t0 = time.time()
    s2_prefix = merge_block(s1, s2, "key_prefix3")
    print(f"  -> key_prefix3 pairs found ({time.time()-t0:.1f}s)", flush=True)
    
    # FREE MEMORY before loading the next massive file
    del s2
    gc.collect()
    
    print("\n--- Blocking against Source 3 ---", flush=True)
    s3 = load_and_normalize(TEST_SOURCE3, nrows=TESTING_ROWS)
    
    t0 = time.time()
    s3_key2 = merge_block(s1, s3, "key_2tokens")
    print(f"  -> key_2tokens pairs found ({time.time()-t0:.1f}s)", flush=True)
    
    t0 = time.time()
    s3_prefix = merge_block(s1, s3, "key_prefix3")
    print(f"  -> key_prefix3 pairs found ({time.time()-t0:.1f}s)", flush=True)
    
    # FREE MEMORY
    del s3
    gc.collect()
    
    print("\nCombining and capping candidates per entity...", flush=True)
    merged = union_and_cap(s2_key2, s2_prefix, s3_key2, s3_prefix)
    
    rows = []
    for eid in s1["entity_id"]:
        candidates = merged.get(eid, set())
        rows.append({
            "source1_entity_id": eid,
            "candidate_entity_ids": ",".join(sorted(candidates)),
        })
    result_df = pd.DataFrame(rows)
    
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