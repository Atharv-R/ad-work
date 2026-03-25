# scripts/train_ctr.py

"""
Train CTR prediction model on Criteo Click Logs.

Usage:
    # Full pipeline (downloads Criteo if needed)
    uv run python scripts/train_ctr.py

    # Quick test with synthetic data
    uv run python scripts/train_ctr.py --synthetic

    # Custom settings
    uv run python scripts/train_ctr.py --sample-size 500000 --data-path data/raw/criteo/day_0.tsv.gz

Windows:
    .venv\\Scripts\\python scripts\\train_ctr.py
    .venv\\Scripts\\python scripts\\train_ctr.py --synthetic
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from loguru import logger


DEFAULT_DATA_PATH = Path("data/raw/criteo/day_0.tsv.gz")
DOWNLOAD_URL = "https://huggingface.co/datasets/reczilla/criteo-click-logs/resolve/main/day_0.tsv.gz"


def download_criteo(dest: Path):
    """Download one day of Criteo data from HuggingFace."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Criteo Click Logs (~1.4 GB)...")
    print(f"URL: {DOWNLOAD_URL}")
    print(f"Destination: {dest}")
    print("This may take several minutes depending on your connection.\n")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(DOWNLOAD_URL, str(dest), reporthook=progress_hook)
    print(f"\n\nDownload complete: {dest}")


def main(args):
    from sklearn.model_selection import train_test_split
    from adwork.models.ctr_model import CriteoDataLoader, CriteoPreprocessor, CTRModel
    from adwork.models.registry import save_ctr_artifacts

    print("=" * 60)
    print("  Ad-Work CTR Model Training")
    print("=" * 60)
    print()

    # ── Step 1: Load data ──
    if args.synthetic:
        print("[1/5] Generating synthetic data (for pipeline testing only)...")
        df = CriteoDataLoader.generate_synthetic(n_rows=args.sample_size)
        data_source = "synthetic"
    else:
        data_path = Path(args.data_path)

        if not data_path.exists():
            print(f"[!] Criteo data not found at {data_path}")
            response = input("    Download it now? (~1.4 GB) [y/N]: ").strip().lower()
            if response == "y":
                download_criteo(data_path)
            else:
                print("    Use --synthetic flag for quick testing:")
                print("    uv run python scripts/train_ctr.py --synthetic")
                return

        print(f"[1/5] Loading Criteo data ({args.sample_size:,} samples)...")
        df = CriteoDataLoader.load(data_path, sample_size=args.sample_size)
        data_source = str(data_path)

    print(f"       Loaded {len(df):,} rows | Click rate: {df['label'].mean():.4f}")
    print()

    # ── Step 2: Preprocess ──
    print("[2/5] Preprocessing features...")
    preprocessor = CriteoPreprocessor(max_cardinality=args.max_cardinality)
    X, y = preprocessor.fit_transform(df)

    # Free memory
    del df

    # Split: 70% train / 15% val / 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval,
        # 0.176 of 0.85 ≈ 0.15 of total
    )

    print(f"       Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print()

    # ── Step 3: Train ──
    print("[3/5] Training LightGBM...")
    model = CTRModel()
    train_result = model.train(
        X_train, y_train, X_val, y_val,
        categorical_features=preprocessor.categorical_feature_names,
        num_boost_round=args.num_rounds,
        early_stopping_rounds=50,
    )
    print(f"       Best iteration: {train_result['best_iteration']}")
    print(f"       Val log loss:   {train_result['best_val_logloss']:.4f}")
    print()

    # ── Step 4: Evaluate ──
    print("[4/5] Evaluating on test set...")
    evaluation = model.evaluate(X_test, y_test)

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │         CTR Model Evaluation             │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  AUC-ROC:           {evaluation.auc_roc:.4f}              │")
    print(f"  │  Log Loss:          {evaluation.log_loss:.4f}              │")
    print(f"  │  PR-AUC:            {evaluation.pr_auc:.4f}              │")
    print(f"  │  Calibration Error: {evaluation.calibration_error:.4f}              │")
    print(f"  │  Base Rate:         {evaluation.base_rate:.4f}              │")
    print(f"  │  Test Samples:      {evaluation.test_size:,}            │")
    print("  └─────────────────────────────────────────┘")
    print()

    # Feature importance
    print("  Top 10 Features by Importance (Gain):")
    for i, (feat, imp) in enumerate(list(evaluation.feature_importance.items())[:10]):
        bar = "█" * int(imp / max(evaluation.feature_importance.values()) * 20)
        print(f"    {i+1:>2}. {feat:<6} {bar} {imp:.0f}")
    print()

    # ── Step 5: Save ──
    print("[5/5] Saving model artifacts...")
    save_ctr_artifacts(
        model=model,
        preprocessor=preprocessor,
        evaluation=evaluation,
        extra_metadata={
            "data_source": data_source,
            "sample_size": args.sample_size,
            "max_cardinality": args.max_cardinality,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        },
    )

    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Model saved to: models/ctr/")
    print(f"  AUC-ROC: {evaluation.auc_roc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTR model on Criteo data")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for quick pipeline testing",
    )
    parser.add_argument(
        "--data-path", type=str, default=str(DEFAULT_DATA_PATH),
        help="Path to Criteo TSV file",
    )
    parser.add_argument(
        "--sample-size", type=int, default=1_000_000,
        help="Number of rows to sample from Criteo data (default: 1M)",
    )
    parser.add_argument(
        "--max-cardinality", type=int, default=10_000,
        help="Max unique values per categorical feature (default: 10K)",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=500,
        help="Max boosting rounds (default: 500, early stopping may stop earlier)",
    )

    main(parser.parse_args())