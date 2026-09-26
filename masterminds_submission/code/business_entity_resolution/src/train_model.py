import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, 
    TRAIN_GROUND_TRUTH, RANDOM_SEED
)
from blocking import load_and_normalize, build_candidate_pairs
from features import build_pairwise_features, FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
THRESHOLD_PATH = Path(__file__).resolve().parent / "threshold.json"

def main():
    s1 = load_and_normalize(TRAIN_SOURCE1)
    s2 = load_and_normalize(TRAIN_SOURCE2)
    s3 = load_and_normalize(TRAIN_SOURCE3)
    other = pd.concat([s2, s3], ignore_index=True)

    gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", dtype=str).fillna("")
    
    gt_exploded = gt.assign(
        matched_entity_id=gt['matched_entity_ids'].str.split(',')
    ).explode('matched_entity_id')
    gt_exploded = gt_exploded[gt_exploded['matched_entity_id'] != ""]
    true_pairs = set(zip(gt_exploded['source1_entity_id'], gt_exploded['matched_entity_id']))

    train_candidate_pairs = build_candidate_pairs(s1, s2, s3)
    features_df = build_pairwise_features(s1, other, train_candidate_pairs)

    features_df['is_match'] = features_df.apply(
        lambda x: 1 if (x['source1_entity_id'], x['candidate_entity_id']) in true_pairs else 0,
        axis=1
    )

    X = features_df[FEATURE_COLUMNS]
    y = features_df['is_match']

    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=RANDOM_SEED, 
        n_jobs=-1
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({"best_threshold": 0.50}))

if __name__ == "__main__":
    main()