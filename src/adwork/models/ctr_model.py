# src/adwork/models/ctr_model.py

"""
CTR Prediction Model
====================
LightGBM-based click-through rate prediction trained on Criteo Click Logs.

Criteo data format (TSV, no header):
- Column 0:     label (1 = click, 0 = no click)
- Columns 1-13: integer features (I1–I13), anonymized counts/aggregates
- Columns 14-39: categorical features (C1–C26), hashed hex strings

Pipeline:
1. Load & sample Criteo data
2. Preprocess: handle NaN, cap cardinality, encode categoricals
3. Train LightGBM with early stopping
4. Evaluate: AUC-ROC, Log Loss, PR-AUC, Calibration

The trained model feeds into the contextual bandit (Phase 4)
for bid optimization under uncertainty.

Usage:
    loader = CriteoDataLoader()
    df = loader.load("data/raw/criteo/day_0.tsv.gz", sample_size=1_000_000)

    preprocessor = CriteoPreprocessor()
    X, y = preprocessor.fit_transform(df)

    model = CTRModel()
    model.train(X_train, y_train, X_val, y_val)
    metrics = model.evaluate(X_test, y_test)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field


# ─── Column Definitions ──────────────────────────────

LABEL_COL = "label"
NUM_COLS = [f"I{i}" for i in range(1, 14)]       # I1–I13  (13 features)
CAT_COLS = [f"C{i}" for i in range(1, 27)]       # C1–C26  (26 features)
ALL_COLS = [LABEL_COL] + NUM_COLS + CAT_COLS      # 40 total


# ─── Evaluation Results Schema ────────────────────────

class CTREvaluation(BaseModel):
    """Structured evaluation results for the CTR model."""
    auc_roc: float = Field(description="Area Under ROC Curve")
    log_loss: float = Field(description="Binary Cross-Entropy Loss")
    pr_auc: float = Field(description="Area Under Precision-Recall Curve")
    calibration_error: float = Field(description="Expected Calibration Error")
    base_rate: float = Field(description="Positive class rate in test set")
    test_size: int = Field(description="Number of test samples")
    calibration_curve: dict = Field(default={}, description="Binned calibration data")
    feature_importance: dict = Field(default={}, description="Top features by gain")


# ─── Data Loader ──────────────────────────────────────

class CriteoDataLoader:
    """
    Load and sample Criteo Click Logs.
    
    Handles the raw TSV format (no headers, tab-separated,
    missing values as empty strings).
    """

    @staticmethod
    def load(
        filepath: str | Path,
        sample_size: int = 1_000_000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Load Criteo data from a .tsv or .tsv.gz file.

        Reads more rows than needed, then samples for representativeness.
        
        Args:
            filepath: Path to day_X.tsv.gz file
            sample_size: Target number of rows
            random_state: For reproducible sampling
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Criteo data not found at {filepath}\n"
                f"Download it first: uv run python scripts/download_criteo.py"
            )

        # Read more than needed so sampling is representative
        read_size = min(sample_size * 3, 15_000_000)

        logger.info(f"Loading Criteo data from {filepath.name} (reading {read_size:,} rows)...")

        compression = "gzip" if filepath.suffix == ".gz" else None

        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=ALL_COLS,
            nrows=read_size,
            na_values=[""],
            compression=compression,
            low_memory=False,
        )

        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size:,} rows from {read_size:,} loaded")
        else:
            logger.info(f"Loaded {len(df):,} rows (full file)")

        logger.info(
            f"Click rate: {df[LABEL_COL].mean():.4f} "
            f"({df[LABEL_COL].sum():,} clicks / {len(df):,} impressions)"
        )

        return df.reset_index(drop=True)

    @staticmethod
    def generate_synthetic(n_rows: int = 200_000, random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic Criteo-like data for pipeline testing.
        NOT for model evaluation — use real Criteo data for that.
        """
        rng = np.random.default_rng(random_state)
        logger.warning("Generating SYNTHETIC data — use real Criteo for evaluation")

        data = {LABEL_COL: rng.binomial(1, 0.03, n_rows)}  # ~3% CTR

        for col in NUM_COLS:
            vals = rng.exponential(scale=5.0, size=n_rows).astype(int)
            mask = rng.random(n_rows) < 0.15  # 15% missing
            vals = vals.astype(float)
            vals[mask] = np.nan
            data[col] = vals

        for col in CAT_COLS:
            cardinality = rng.integers(100, 50000)
            vocab = [f"{rng.integers(0, 2**32):08x}" for _ in range(cardinality)]
            indices = rng.integers(0, len(vocab), n_rows)
            vals = [vocab[i] for i in indices]
            mask = rng.random(n_rows) < 0.10
            for i in range(n_rows):
                if mask[i]:
                    vals[i] = None
            data[col] = vals

        return pd.DataFrame(data)


# ─── Preprocessor ─────────────────────────────────────

class CriteoPreprocessor:
    """
    Feature engineering for Criteo data.
    
    Numerical features: Fill NaN with 0, apply log1p transform.
    Categorical features: Cap cardinality, label-encode as integers.
    
    Follows sklearn-style fit/transform pattern so the same
    transformations are applied at prediction time.
    """

    def __init__(self, max_cardinality: int = 10_000):
        self.max_cardinality = max_cardinality
        self._cat_maps: dict[str, dict] = {}   # col → {value: int_code}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Fit on training data and return (X, y).
        
        Args:
            df: Raw Criteo DataFrame with label + features
            
        Returns:
            (X, y) where X is feature-engineered DataFrame, y is label array
        """
        y = df[LABEL_COL].values.astype(int)
        X = df.drop(columns=[LABEL_COL]).copy()

        # ── Numerical: fill NaN, log1p ──
        for col in NUM_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0).astype(float)
                X[col] = np.log1p(np.abs(X[col]))

        # ── Categorical: build vocab, encode ──
        for col in CAT_COLS:
            if col not in X.columns:
                continue

            X[col] = X[col].fillna("__MISSING__").astype(str)

            # Count frequencies
            counts = X[col].value_counts()

            # Keep top K values
            top_values = counts.head(self.max_cardinality).index.tolist()

            # Build mapping: value → integer code
            # 0 = rare/unknown, 1..K = known values
            mapping = {val: idx + 1 for idx, val in enumerate(top_values)}
            self._cat_maps[col] = mapping

            # Apply mapping
            X[col] = X[col].map(lambda v: mapping.get(v, 0)).astype(int)

        self._fitted = True
        logger.info(
            f"Preprocessor fitted: {len(NUM_COLS)} numeric + {len(CAT_COLS)} categorical features, "
            f"max cardinality capped at {self.max_cardinality}"
        )

        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        """
        Transform new data using fitted mappings.
        Returns (X, y) if label exists, else (X, None).
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted — call fit_transform first")

        has_label = LABEL_COL in df.columns
        y = df[LABEL_COL].values.astype(int) if has_label else None
        X = df.drop(columns=[LABEL_COL], errors="ignore").copy()

        for col in NUM_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0).astype(float)
                X[col] = np.log1p(np.abs(X[col]))

        for col in CAT_COLS:
            if col not in X.columns:
                continue
            mapping = self._cat_maps.get(col, {})
            X[col] = X[col].fillna("__MISSING__").astype(str)
            X[col] = X[col].map(lambda v, m=mapping: m.get(v, 0)).astype(int)

        return X, y

    @property
    def categorical_feature_names(self) -> list[str]:
        """Column names for categorical features (for LightGBM)."""
        return [col for col in CAT_COLS if col in self._cat_maps]


# ─── CTR Model ────────────────────────────────────────

class CTRModel:
    """
    LightGBM CTR prediction model.
    
    Architecture choices:
    - GBDT (gradient boosted decision trees) over neural nets
      because tabular data + mixed feature types + interpretability
    - 127 leaves balances complexity vs overfitting for ~1M samples
    - Early stopping on validation log loss prevents overfitting
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.best_iteration = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        categorical_features: list[str] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> dict:
        """
        Train with early stopping on validation set.
        
        Returns:
            Training summary dict
        """
        import lightgbm as lgb

        cat_features = categorical_features or []

        dtrain = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=cat_features if cat_features else "auto",
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val, label=y_val,
            reference=dtrain,
            categorical_feature=cat_features if cat_features else "auto",
            free_raw_data=False,
        )

        logger.info(
            f"Training LightGBM: {X_train.shape[0]:,} train / {X_val.shape[0]:,} val / "
            f"{X_train.shape[1]} features / {len(cat_features)} categorical"
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=50),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            num_boost_round=num_boost_round,
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration

        logger.info(f"Training complete: best iteration = {self.best_iteration}")

        return {
            "best_iteration": self.best_iteration,
            "best_val_logloss": self.model.best_score["val"]["binary_logloss"],
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict click probability for each sample."""
        if self.model is None:
            raise RuntimeError("Model not trained — call train() first")
        return self.model.predict(X, num_iteration=self.best_iteration)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> CTREvaluation:
        """
        Comprehensive evaluation on held-out test set.
        
        Computes:
        - AUC-ROC (discrimination ability)
        - Log Loss (probabilistic accuracy)
        - PR-AUC (important for imbalanced data)
        - Expected Calibration Error (are probabilities reliable?)
        - Calibration curve data (for plotting)
        - Feature importance (for interpretability)
        """
        from sklearn.metrics import (
            roc_auc_score,
            log_loss,
            average_precision_score,
        )

        from sklearn.calibration import calibration_curve

        y_pred = self.predict_proba(X_test)

        # Core metrics
        auc = roc_auc_score(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_pred)

        # Calibration
        n_bins = 10
        fraction_pos, mean_predicted = calibration_curve(
            y_test, y_pred, n_bins=n_bins, strategy="uniform",
        )

        # Expected Calibration Error
        bin_counts = np.histogram(y_pred, bins=n_bins, range=(0, 1))[0]
        total = len(y_test)
        ece = sum(
            (bin_counts[i] / total) * abs(fraction_pos[i] - mean_predicted[i])
            for i in range(len(fraction_pos))
            if bin_counts[i] > 0
        )

        # Feature importance
        importance = self.model.feature_importance(importance_type="gain")
        feature_names = self.model.feature_name()
        feat_imp = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1], reverse=True,
        )

        evaluation = CTREvaluation(
            auc_roc=round(auc, 4),
            log_loss=round(ll, 4),
            pr_auc=round(pr_auc, 4),
            calibration_error=round(ece, 4),
            base_rate=round(float(y_test.mean()), 4),
            test_size=len(y_test),
            calibration_curve={
                "fraction_positive": [round(float(v), 4) for v in fraction_pos],
                "mean_predicted": [round(float(v), 4) for v in mean_predicted],
            },
            feature_importance={name: round(float(imp), 2) for name, imp in feat_imp[:15]},
        )

        logger.info(
            f"Evaluation: AUC-ROC={auc:.4f} | LogLoss={ll:.4f} | "
            f"PR-AUC={pr_auc:.4f} | ECE={ece:.4f}"
        )

        return evaluation

    def save_model(self, filepath: str | Path):
        """Save the LightGBM model to disk."""
        if self.model is None:
            raise RuntimeError("No trained model to save")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str | Path):
        """Load a LightGBM model from disk."""
        import lightgbm as lgb
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = lgb.Booster(model_file=str(filepath))
        self.best_iteration = self.model.best_iteration
        logger.info(f"Model loaded from {filepath}")