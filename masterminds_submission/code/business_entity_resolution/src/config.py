"""
config.py — single source of truth for paths, shared across all pipeline
scripts. Import from here instead of hardcoding relative paths in every file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root regardless of which script imports this file.
# This file lives at:
#   masterminds_project/masterminds_submission/code/business_entity_resolution/src/config.py
# so PROJECT_ROOT is 4 levels up from here.
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]          # -> masterminds_project/
SUBMISSION_ROOT = THIS_FILE.parents[3]       # -> masterminds_submission/

# ---------------------------------------------------------------------------
# Data locations (given to you, outside the submission folder)
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

UTILS_DIR = PROJECT_ROOT / "utils"
VALIDATE_SCRIPT = UTILS_DIR / "validate_submission.py"

# ---------------------------------------------------------------------------
# Output locations (inside the submission folder — these get zipped)
# ---------------------------------------------------------------------------
OUTPUT_DIR = SUBMISSION_ROOT / "output"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# ---------------------------------------------------------------------------
# Blocking / matching hyperparameters (tune these as you iterate)
# ---------------------------------------------------------------------------
MAX_CANDIDATES_PER_ENTITY = 50   # cap after ranking, per blocking strategy union
TFIDF_NGRAM_RANGE = (2, 4)       # character n-grams for name/address vectorization
TFIDF_TOP_K = 30                 # nearest neighbors to pull from TF-IDF similarity
VALIDATION_HOLDOUT_FRACTION = 0.15
RANDOM_SEED = 42

if __name__ == "__main__":
    # quick sanity check — run this file directly to confirm paths resolve
    for name, path in {
        "PROJECT_ROOT": PROJECT_ROOT,
        "SUBMISSION_ROOT": SUBMISSION_ROOT,
        "TRAIN_SOURCE1": TRAIN_SOURCE1,
        "TEST_SOURCE1": TEST_SOURCE1,
        "OUTPUT_DIR": OUTPUT_DIR,
    }.items():
        exists = path.exists()
        print(f"{name:20} -> {path}  [{'OK' if exists else 'missing'}]")