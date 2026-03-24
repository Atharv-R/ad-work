# scripts/download_criteo.py

"""
Download Criteo Click Logs dataset from HuggingFace.

This dataset contains real display ad click data:
- 24 files, each = one day of Criteo traffic
- Each row: click/no-click label + 13 numerical + 26 categorical features
- Used in Phase 3 to train the CTR prediction model

Run with:
    uv run python scripts/download_criteo.py

    Or on Windows:
    .venv\\Scripts\\python scripts\\download_criteo.py

NOTE: Each day's file is ~1.5 GB. Start with --days 1.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger


def download_criteo(num_days: int = 1, output_dir: str = "data/raw/criteo"):
    """
    Download Criteo Click Logs from HuggingFace.
    
    Args:
        num_days: How many days to download (1-24). Start with 1.
        output_dir: Where to save the files
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Run:\n"
            "  uv add huggingface_hub\n"
            "  or: uv pip install huggingface_hub"
        )
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "reczilla/criteo-click-logs"
    num_days = min(num_days, 24)

    for day in range(num_days):
        filename = f"day_{day}.tsv.gz"
        logger.info(f"Downloading {filename} ({day + 1}/{num_days})...")

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(output_path),
            )
            size_mb = Path(downloaded).stat().st_size / (1024 * 1024)
            logger.info(f"  Saved: {downloaded} ({size_mb:.0f} MB)")

        except Exception as e:
            logger.error(f"  Failed to download {filename}: {e}")
            continue

    logger.info(f"Download complete. Files in: {output_path}")
    logger.info("These will be used in Phase 3 for CTR model training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Criteo Click Logs")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to download (1-24). Each day is ~1.5GB. Default: 1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/criteo",
        help="Output directory",
    )

    args = parser.parse_args()

    print(f"Downloading {args.days} day(s) of Criteo Click Logs...")
    print(f"Estimated size: ~{args.days * 1.5:.1f} GB")
    print()

    download_criteo(num_days=args.days, output_dir=args.output)