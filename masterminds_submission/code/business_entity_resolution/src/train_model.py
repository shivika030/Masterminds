import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH, RANDOM_SEED, VALIDATION_HOLDOUT_FRACTION,
)
from blocking import load_and_normalize, build_candidate_pairs
from features import build_pairwise_features, attach_labels, FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
THRESHOLD_PATH = Path(__file__).resolve().parent / "threshold.json"


# ---------------------------------------------------------------------------
# F_0.5 — exact formula from the PDF, computed per Source1 entity
# ---------------------------------------------------------------------------
def entity_f_beta(predicted_ids: set, true_ids: set, beta: float = 0.5) -> float:
    if len(true_ids) == 0:
        # singleton: correct empty prediction = 1.0, any predicted match = 0.0
        return 1.0 if len(predicted_ids) == 0 else 0.0
    if len(predicted_ids) == 0:
        return 0.0  # missed everything on a non-singleton
    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    if precision + recall == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f_beta(pred_map: dict, true_map: dict, all_ids: list, beta: float = 0.5) -> float:
    """pred_map / true_map: {source1_entity_id: set(matched_ids)}. Averages over all_ids."""
    scores = [
        entity_f_beta(pred_map.get(eid, set()), true_map.get(eid, set()), beta)
        for eid in all_ids
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading + normalizing train sources...")
    s1 = load_and_normalize(TRAIN_SOURCE1)
    s2 = load_and_normalize(TRAIN_SOURCE2)
    s3 = load_and_normalize(TRAIN_SOURCE3)
    other = pd.concat([s2, s3], ignore_index=True)

    gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", dtype=str).fillna("")
    true_map = {
        row["source1_entity_id"]: set(row["matched_entity_ids"].split(","))
        if row["matched_entity_ids"] else set()
        for _, row in gt.iterrows()
    }

    # -----------------------------------------------------------------
    # Split by SOURCE1 ENTITY ID (not by pair) — avoids leakage
    # -----------------------------------------------------------------
    train_ids, val_ids = train_test_split(
        s1["entity_id"].tolist(),
        test_size=VALIDATION_HOLDOUT_FRACTION,
        random_state=RANDOM_SEED,
    )
    train_ids, val_ids = set(train_ids), set(val_ids)
    print(f"Train entities: {len(train_ids)} | Validation entities: {len(val_ids)}")

    # -----------------------------------------------------------------
    # Blocking + features on the FULL train set (blocking behaves the
    # same regardless of split — this mirrors what happens at test time)
    # -----------------------------------------------------------------
    print("Running blocking on train sources...")
    candidate_pairs = build_candidate_pairs(s1, s2, s3)

    print("Building pairwise features...")
    features_df = build_pairwise_features(s1, other, candidate_pairs)
    features_df = attach_labels(features_df, gt)  # adds "label" column

    # -----------------------------------------------------------------
    # Blocking recall check on validation entities (recall ceiling)
    # -----------------------------------------------------------------
    val_true_pair_count = sum(len(true_map.get(eid, set())) for eid in val_ids)
    val_candidate_pairs_found = features_df[
        features_df["source1_entity_id"].isin(val_ids) & (features_df["label"] == 1)
    ].shape[0]
    blocking_recall = (
        val_candidate_pairs_found / val_true_pair_count if val_true_pair_count > 0 else float("nan")
    )
    print(f"Blocking recall on validation set: {blocking_recall:.3f} "
          f"({val_candidate_pairs_found}/{val_true_pair_count} true matches survived blocking)")
    if blocking_recall < 0.9:
        print("WARNING: blocking recall is low — your final F_0.5 is capped no matter "
              "how good the matcher is. Revisit blocking.py before tuning the model further.")

    # -----------------------------------------------------------------
    # Train / validation pair split (mirrors the entity split above)
    # -----------------------------------------------------------------
    train_pairs = features_df[features_df["source1_entity_id"].isin(train_ids)]
    val_pairs = features_df[features_df["source1_entity_id"].isin(val_ids)].copy()

    X_train, y_train = train_pairs[FEATURE_COLUMNS], train_pairs["label"]
    print(f"Training pairs: {len(X_train)} (positives: {y_train.sum()}, "
          f"negatives: {len(y_train) - y_train.sum()})")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",  # candidate pairs are heavily skewed toward non-matches
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # Score validation pairs, sweep thresholds, pick the one maximizing F_0.5
    # -----------------------------------------------------------------
    if len(val_pairs) > 0:
        val_pairs["prob"] = model.predict_proba(val_pairs[FEATURE_COLUMNS])[:, 1]
    else:
        val_pairs["prob"] = []

    all_val_ids = list(val_ids)  # includes entities with zero candidates too

    best_threshold, best_score = 0.5, -1.0
    sweep_results = []
    for threshold in np.arange(0.10, 0.96, 0.05):
        pred_map = (
            val_pairs[val_pairs["prob"] >= threshold]
            .groupby("source1_entity_id")["candidate_entity_id"]
            .apply(set)
            .to_dict()
        )
        score = macro_f_beta(pred_map, true_map, all_val_ids, beta=0.5)
        sweep_results.append((round(threshold, 2), round(score, 4)))
        if score > best_score:
            best_score, best_threshold = score, round(threshold, 2)

    print("\nThreshold sweep (threshold, validation F_0.5):")
    for t, s in sweep_results:
        marker = "  <-- best" if t == best_threshold else ""
        print(f"  {t:.2f}: {s:.4f}{marker}")
    print(f"\nBest threshold: {best_threshold} | Validation F_0.5: {best_score:.4f}")

    # -----------------------------------------------------------------
    # Refit on ALL train data (train+val) with the chosen threshold locked in,
    # then save. This squeezes a bit more signal into the final model
    # while the threshold — chosen on held-out data — stays trustworthy.
    # -----------------------------------------------------------------
    print("\nRefitting final model on full training data...")
    X_full, y_full = features_df[FEATURE_COLUMNS], features_df["label"]
    final_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    final_model.fit(X_full, y_full)

    joblib.dump(final_model, MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({"best_threshold": best_threshold}))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved threshold to {THRESHOLD_PATH}")

    # feature importances — useful for your Documentation_template.md writeup
    importances = sorted(
        zip(FEATURE_COLUMNS, final_model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:28} {imp:.4f}")


if __name__ == "__main__":
    main()