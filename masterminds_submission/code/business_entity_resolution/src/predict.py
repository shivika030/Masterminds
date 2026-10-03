import json
from pathlib import Path
import joblib
import pandas as pd

from config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    CANDIDATE_PAIRS_PATH, MATCHING_RESULTS_PATH, OUTPUT_DIR,
)
from normalizer import normalize_name_series, normalize_address_series
from features import build_pairwise_features, FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
THRESHOLD_PATH = Path(__file__).resolve().parent / "threshold.json"

def load_for_features(path: Path) -> pd.DataFrame:
    print(f"  reading {path} ...", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    
    name_info = normalize_name_series(df["business_name"])
    df["clean_name"] = name_info["clean_name"]
    df["clean_address"] = normalize_address_series(df["business_address"])
    df["country"] = df["country"].str.lower().str.strip()
    
    return df[["entity_id", "clean_name", "clean_address", "country"]]

def load_model_and_threshold():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found")
    model = joblib.load(MODEL_PATH)

    if THRESHOLD_PATH.exists():
        threshold = json.loads(THRESHOLD_PATH.read_text())["best_threshold"]
    else:
        threshold = 0.5
    return model, threshold

def main():
    print("Loading datasets for feature extraction...", flush=True)
    s1 = load_for_features(TEST_SOURCE1)
    s2 = load_for_features(TEST_SOURCE2)
    s3 = load_for_features(TEST_SOURCE3)
    other = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    if not CANDIDATE_PAIRS_PATH.exists():
        raise FileNotFoundError(f"{CANDIDATE_PAIRS_PATH} not found")
    print("Loading candidate pairs...", flush=True)
    candidate_pairs = pd.read_csv(CANDIDATE_PAIRS_PATH, sep="\t", dtype=str).fillna("")

    print("Building pairwise features...", flush=True)
    pairs = build_pairwise_features(s1, other, candidate_pairs)
    
    print("Loading model and generating predictions...", flush=True)
    model, threshold = load_model_and_threshold()

    if len(pairs) > 0:
        pairs["match_probability"] = model.predict_proba(pairs[FEATURE_COLUMNS])[:, 1]
        pairs["predicted_match"] = (pairs["match_probability"] >= threshold).astype(int)
    else:
        pairs["match_probability"] = []
        pairs["predicted_match"] = []

    print("Formatting final matching results...", flush=True)
    matched = pairs[pairs["predicted_match"] == 1]
    grouped = (
        matched.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )

    all_s1_ids = pd.DataFrame({"source1_entity_id": s1["entity_id"]})
    result = all_s1_ids.merge(grouped, on="source1_entity_id", how="left")
    result["matched_entity_ids"] = result["matched_entity_ids"].fillna("")

    assert result["source1_entity_id"].is_unique
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(MATCHING_RESULTS_PATH, sep="\t", index=False)
    print(f"Done! Wrote {len(result)} rows to {MATCHING_RESULTS_PATH}", flush=True)

if __name__ == "__main__":
    main()