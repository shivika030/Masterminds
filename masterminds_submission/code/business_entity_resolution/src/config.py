from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
SUBMISSION_ROOT = THIS_FILE.parents[3]

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

OUTPUT_DIR = SUBMISSION_ROOT / "output"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

MAX_CANDIDATES_PER_ENTITY = 50   # cap after ranking, per blocking strategy union
MAX_BLOCK_SIZE = 2000            # a key matching MORE candidates than this is purged
                                  # (too generic — almost certainly noise, not real matches)
S1_CHUNK_SIZE = 100_000          # rows of Source1 processed per merge batch (memory safety)
VALIDATION_HOLDOUT_FRACTION = 0.15
RANDOM_SEED = 42