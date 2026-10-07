"""
blocking.py — candidate generation (blocking) stage, rebuilt for LARGE
datasets (millions of rows per source).

Why this version is different from a naive approach:
  - No Python-level row loops (no .iterrows(), no per-row .apply() for
    matching). Everything uses pandas merge (a hash join under the hood),
    which scales to millions of rows in seconds, not hours.
  - TF-IDF + NearestNeighbors comparing every Source1 row against millions
    of candidates directly is NOT used here — it's computationally
    infeasible at this scale on a single machine. If recall from key-based
    blocking isn't high enough, the fix is smarter/more blocking KEYS
    (still merge-based), not brute-force similarity search.

Output: candidate_pairs.tsv with columns
  source1_entity_id \t candidate_entity_ids (comma-separated, S2-/S3- only)

Run directly to generate candidates for the TEST set:
    python3 blocking.py
"""
import time
from collections import defaultdict

import pandas as pd

from config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    CANDIDATE_PAIRS_PATH, OUTPUT_DIR,
    MAX_CANDIDATES_PER_ENTITY, MAX_BLOCK_SIZE, S1_CHUNK_SIZE, MAX_CHUNK_MERGE_ROWS,
)
from normalizer import normalize_name_series, normalize_address_series


# ---------------------------------------------------------------------------
# 1. Load + normalize one source file (vectorized — fast even at millions
#    of rows). Also builds several BLOCKING KEY columns used for merging.
# ---------------------------------------------------------------------------
def load_and_normalize(path) -> pd.DataFrame:
    t0 = time.time()
    print(f"  reading {path} ...", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    print(f"    {len(df)} rows, normalizing names...", flush=True)

    name_info = normalize_name_series(df["business_name"])
    df["clean_name"] = name_info["clean_name"]
    df["full_normalized_name"] = name_info["full_normalized"]

    print(f"    normalizing addresses...", flush=True)
    df["clean_address"] = normalize_address_series(df["business_address"])

    country_norm = df["country"].str.lower().str.strip()

    # Key 1: first two tokens of the clean name + country.
    # Reasonably selective — catches exact/near-exact name matches.
    df["key_2tokens"] = (
        df["clean_name"].str.split().str[:2].str.join(" ") + "|" + country_norm
    )

    # Key 2: first FIVE characters of the name (no spaces) + country.
    # Uses 5 chars rather than 3 — at millions-of-rows scale, a 3-char
    # prefix has too few distinct values (business names cluster onto a
    # small number of common starting letters), causing every block to be
    # uniformly oversized rather than a few outliers purging can catch.
    # 5 characters gives far more distinct combinations and much smaller,
    # genuinely useful blocks.
    df["key_prefix5"] = (
        df["clean_name"].str.replace(" ", "", regex=False).str[:5] + "|" + country_norm
    )

    print(f"    done: {path}  ({time.time()-t0:.1f}s)", flush=True)
    return df


# ---------------------------------------------------------------------------
# 2. Merge-based blocking on ONE key column — vectorized hash join, with
#    BLOCK PURGING and CHUNKING for safety at scale.
#
#    Why purging is necessary: if a small fraction of records share an
#    overly generic key value (e.g. a short/common name fragment), that ONE
#    key can match thousands of records on each side. The merge then
#    produces a combinatorial explosion (e.g. 5% collision on a couple of
#    keys can turn a 200K x 500K join into 125M+ rows) that exhausts memory
#    before you even get to grouping — and those huge blocks are almost
#    never genuine matches anyway, just noise. So before merging, any key
#    value that matches MORE than max_block_size candidates on the "other"
#    side is dropped entirely — it wasn't going to produce a useful
#    candidate set regardless.
#
#    Chunking s1 on top of that bounds peak memory further, in case any
#    single surviving block is still large.
# ---------------------------------------------------------------------------
def merge_block(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, key_col: str,
    max_block_size: int = MAX_BLOCK_SIZE, max_s1_chunk_size: int = S1_CHUNK_SIZE,
    target_chunk_merge_rows: int = MAX_CHUNK_MERGE_ROWS,
) -> dict:
    # --- purge oversized blocks on the "other" side ---
    key_counts = other_df[key_col].value_counts()
    oversized_keys = key_counts[key_counts > max_block_size]
    if len(oversized_keys) > 0:
        print(f"    purging {len(oversized_keys)} oversized key(s) "
              f"(covering {oversized_keys.sum()} rows, e.g. {oversized_keys.index[0]!r}: "
              f"{oversized_keys.iloc[0]} matches) — too generic to be useful blocking", flush=True)
    surviving_counts = key_counts[key_counts <= max_block_size]
    valid_keys = set(surviving_counts.index)

    other_small = other_df[other_df[key_col].isin(valid_keys)][["entity_id", key_col]].rename(
        columns={"entity_id": "candidate_id"}
    )
    if len(other_small) == 0:
        return {}

    # --- ADAPTIVE chunk sizing: even after purging, a key can still be
    #     uniformly non-selective (many keys just under the purge cap,
    #     rather than a few extreme outliers) — a fixed chunk size can't
    #     protect against that. Instead, size each chunk so that its
    #     worst-case merge blowup (chunk_size x largest surviving block)
    #     never exceeds a fixed, memory-safe row budget.
    max_remaining_block = int(surviving_counts.max()) if len(surviving_counts) > 0 else 1
    adaptive_chunk_size = max(1, min(max_s1_chunk_size, target_chunk_merge_rows // max_remaining_block))
    if adaptive_chunk_size < max_s1_chunk_size:
        print(f"    largest surviving block: {max_remaining_block} — "
              f"shrinking chunk size to {adaptive_chunk_size} for memory safety", flush=True)

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


# ---------------------------------------------------------------------------
# 3. Combine multiple key strategies (union) and cap per entity.
#    IMPORTANT: naive truncation of a set (list(matches)[:cap]) is
#    essentially arbitrary — Python set ordering has no relationship to
#    match quality. Instead, we count how many blocking STRATEGIES found
#    each candidate (agreement across 2+ keys is a real signal of a better
#    match) and keep the highest-agreement candidates first, capping only
#    after that ranking. Ties break alphabetically for determinism.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Main — build candidates against BOTH Source 2 and Source 3
# ---------------------------------------------------------------------------
def build_candidate_pairs(s1_df, s2_df, s3_df) -> pd.DataFrame:
    print("Blocking against Source 2 (key_2tokens)...", flush=True)
    t0 = time.time()
    s2_key2 = merge_block(s1_df, s2_df, "key_2tokens")
    print(f"  -> {sum(len(v) for v in s2_key2.values())} pairs found ({time.time()-t0:.1f}s)", flush=True)

    print("Blocking against Source 2 (key_prefix5)...", flush=True)
    t0 = time.time()
    s2_prefix = merge_block(s1_df, s2_df, "key_prefix5")
    print(f"  -> {sum(len(v) for v in s2_prefix.values())} pairs found ({time.time()-t0:.1f}s)", flush=True)

    print("Blocking against Source 3 (key_2tokens)...", flush=True)
    t0 = time.time()
    s3_key2 = merge_block(s1_df, s3_df, "key_2tokens")
    print(f"  -> {sum(len(v) for v in s3_key2.values())} pairs found ({time.time()-t0:.1f}s)", flush=True)

    print("Blocking against Source 3 (key_prefix5)...", flush=True)
    t0 = time.time()
    s3_prefix = merge_block(s1_df, s3_df, "key_prefix5")
    print(f"  -> {sum(len(v) for v in s3_prefix.values())} pairs found ({time.time()-t0:.1f}s)", flush=True)

    print("Combining and capping candidates per entity...", flush=True)
    merged = union_and_cap(s2_key2, s2_prefix, s3_key2, s3_prefix)

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
    print("Loading and normalizing test sources...", flush=True)
    s1 = load_and_normalize(TEST_SOURCE1)
    s2 = load_and_normalize(TEST_SOURCE2)
    s3 = load_and_normalize(TEST_SOURCE3)

    print(f"Source1: {len(s1)} | Source2: {len(s2)} | Source3: {len(s3)}", flush=True)

    result_df = build_candidate_pairs(s1, s2, s3)

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