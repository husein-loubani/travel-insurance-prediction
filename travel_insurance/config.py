"""
config.py
---------
Global constants, feature lists, color palette, and model hyperparameter grids
for the Travel Insurance Prediction project.

All magic numbers and strings are centralised here so the notebook contains no
hardcoded literals and changes propagate to every cell automatically.
"""

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Statistical significance threshold ───────────────────────────────────────
ALPHA            = 0.05
CONFIDENCE_LEVEL = 0.95

# ── Evaluation metrics ────────────────────────────────────────────────────────
PRIMARY_METRIC   = "roc_auc"   # robust to class imbalance; use for CV and ranking
SECONDARY_METRICS = ["pr_auc", "f1", "balanced_accuracy"]

# ── Data ─────────────────────────────────────────────────────────────────────
TARGET = "travel_insurance"
INDEX_COL = "Unnamed: 0"   # row-index column in the raw CSV

# Feature groups
CATEGORICAL_FEATURES = [
    "employment_type",
    "graduate_or_not",
    "frequent_flyer",
    "ever_travelled_abroad",
    "chronic_diseases",      # binary int (0/1); treated as categorical, passthrough in pipeline
]

BINARY_STRING_MAP = {
    "graduate_or_not":       {"Yes": 1, "No": 0},
    "frequent_flyer":       {"Yes": 1, "No": 0},
    "ever_travelled_abroad": {"Yes": 1, "No": 0},
}

EMPLOYMENT_CATEGORIES = ["Government Sector", "Private Sector/Self Employed"]

NUMERICAL_FEATURES = [
    "age",
    "annual_income",
    "family_members",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ── Train / test split ───────────────────────────────────────────────────────
TEST_SIZE = 0.20   # 80/20 stratified split; test set is held out until final evaluation

# ── Color palette ───────────────────────────────────────────────────────────
# Two-class palette: class 0 (no insurance) → muted blue, class 1 → coral
PALETTE = {0: "#4C72B0", 1: "#DD8452"}
PALETTE_LIST = ["#4C72B0", "#DD8452"]

# Sequential palette for heatmaps
CMAP_DIV  = "RdBu_r"
CMAP_SEQ  = "Blues"

# ── Hyperparameter grids ─────────────────────────────────────────────────────
LR_GRID = {
    "clf__C":       [0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l1", "l2"],
    "clf__solver":  ["liblinear"],
}

NB_GRID = {
    "clf__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
}

KNN_GRID = {
    "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "clf__weights":     ["uniform", "distance"],
    "clf__metric":      ["euclidean", "manhattan"],
}

SVM_GRID = {
    "clf__C":      [0.1, 1, 10, 100],
    "clf__kernel": ["rbf", "linear"],
    "clf__gamma":  ["scale", "auto"],
}

RF_GRID = {
    "clf__n_estimators":      [100, 200, 300],
    "clf__max_depth":         [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__max_features":      ["sqrt", "log2"],
}
