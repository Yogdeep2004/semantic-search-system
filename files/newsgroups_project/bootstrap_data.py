"""
bootstrap_data.py
-----------------
Ensures the 20 Newsgroups dataset is available at data/raw/20_newsgroups.
If not present, downloads via sklearn.datasets.fetch_20newsgroups and
writes the expected folder structure so build_index.py can run without
a manual ZIP download.
"""

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_20_newsgroups(extract_root: str = "./data/raw") -> str:
    """
    If data/raw/20_newsgroups does not exist, fetch the dataset via sklearn
    and write it to extract_root/20_newsgroups in the format expected by
    load_newsgroups_documents (one folder per category, plain text files).

    Returns the path to the 20_newsgroups directory.
    """
    data_dir = os.path.join(extract_root, "20_newsgroups")
    if os.path.isdir(data_dir):
        # Quick sanity check: has category subdirs
        subdirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(subdirs) >= 5:
            logger.info(f"Dataset already present at {data_dir}")
            return data_dir

    logger.info("Downloading 20 Newsgroups via sklearn (one-time) …")
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(subset="all", remove=(), data_home=None)
    # data: list of raw document strings
    # filenames: list of paths (e.g. .../20news-bydate-train/rec.sport.hockey/12345)
    # target: category index; target_names: category names
    data = bunch.data
    target = bunch.target
    target_names = bunch.target_names

    os.makedirs(data_dir, exist_ok=True)
    for i, (text, cat_idx) in enumerate(zip(data, target)):
        category = target_names[cat_idx]
        cat_dir = os.path.join(data_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        # Use a unique filename (sklearn often uses numeric ids)
        out_path = os.path.join(cat_dir, str(i))
        Path(out_path).write_text(text, encoding="utf-8", errors="replace")

    logger.info(f"Wrote {len(data)} documents to {data_dir}")
    return data_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bootstrap 20 Newsgroups dataset.")
    parser.add_argument("--extract_root", default="./data/raw", help="Parent dir for 20_newsgroups")
    args = parser.parse_args()
    bootstrap_20_newsgroups(extract_root=args.extract_root)
