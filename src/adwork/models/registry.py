# src/adwork/models/registry.py

"""
Model Registry
==============
Simple file-based model versioning.

Structure:
    models/
      ctr/
        model.lgb              # LightGBM model
        preprocessor.joblib    # Fitted CriteoPreprocessor
        metadata.json          # Training info, metrics, params
"""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

MODELS_DIR = Path("models")


def save_ctr_artifacts(
    model,           # CTRModel instance
    preprocessor,    # CriteoPreprocessor instance
    evaluation,      # CTREvaluation instance
    extra_metadata: dict | None = None,
) -> Path:
    """
    Save all CTR model artifacts to the registry.
    
    Returns:
        Path to the model directory
    """
    import joblib

    model_dir = MODELS_DIR / "ctr"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM model
    model.save_model(model_dir / "model.lgb")

    # Save preprocessor
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    # Save metadata
    metadata = {
        "model_type": "lightgbm_ctr",
        "trained_at": datetime.now().isoformat(),
        "best_iteration": model.best_iteration,
        "params": model.params,
        "evaluation": evaluation.model_dump(),
        **(extra_metadata or {}),
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"All CTR artifacts saved to {model_dir}")
    return model_dir


def load_ctr_artifacts() -> tuple:
    """
    Load CTR model, preprocessor, and metadata.
    
    Returns:
        (CTRModel, CriteoPreprocessor, metadata_dict)
        
    Raises:
        FileNotFoundError if model hasn't been trained yet
    """
    import joblib

    from adwork.models.ctr_model import CTRModel

    model_dir = MODELS_DIR / "ctr"

    model_path = model_dir / "model.lgb"
    preproc_path = model_dir / "preprocessor.joblib"
    meta_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "CTR model not found. Train it first:\n"
            "  uv run python scripts/train_ctr.py"
        )

    # Load model
    ctr_model = CTRModel()
    ctr_model.load_model(model_path)

    # Load preprocessor
    preprocessor = joblib.load(preproc_path) if preproc_path.exists() else None

    # Load metadata
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    logger.info(f"CTR artifacts loaded from {model_dir}")
    return ctr_model, preprocessor, metadata


def ctr_model_exists() -> bool:
    """Check if a trained CTR model exists."""
    return (MODELS_DIR / "ctr" / "model.lgb").exists()


def get_ctr_metadata() -> dict | None:
    """Get CTR model metadata without loading the full model."""
    meta_path = MODELS_DIR / "ctr" / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)