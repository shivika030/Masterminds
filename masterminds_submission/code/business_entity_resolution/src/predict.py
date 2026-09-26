import json
from pathlib import Path
import joblib
import pandas as pd

from config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    CANDIDATE_PAIRS_PATH, MATCHING_RESULTS_PATH, OUTPUT_DIR,
)
from blocking import load_and_normalize
from features import build_pairwise_features, FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
THRESHOLD_PATH = Path(__file__).resolve().parent / "threshold.json"

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
    s1 = load_and_normalize(TEST_SOURCE1)
    s2 = load_and_normalize(TEST_SOURCE2)
    s3 = load_and_normalize(TEST_SOURCE3)
    other = pd.concat([s2, s3], ignore_index=True)

    if not CANDIDATE_PAIRS_PATH.exists():
        raise FileNotFoundError(f"{CANDIDATE_PAIRS_PATH} not found")
    candidate_pairs = pd.read_csv(CANDIDATE_PAIRS_PATH, sep="\t", dtype=str).fillna("")

    pairs = build_pairwise_features(s1, other, candidate_pairs)
    model, threshold = load_model_and_threshold()

    if len(pairs) > 0:
        pairs["match_probability"] = model.predict_proba(pairs[FEATURE_COLUMNS])[:, 1]
        pairs["predicted_match"] = (pairs["match_probability"] >= threshold).astype(int)
    else:
        pairs["match_probability"] = []
        pairs["predicted_match"] = []

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

if __name__ == "__main__":
    main()