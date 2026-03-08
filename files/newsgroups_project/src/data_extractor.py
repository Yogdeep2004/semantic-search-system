"""
data_extractor.py
-----------------
Dataset Extractor: automated extraction of ZIP and TAR.GZ archives
containing plain text documents from the 20 Newsgroups dataset.

The 20 Newsgroups dataset is distributed as:
  ZIP archive → TAR.GZ archive → plain text documents

Each document is a raw Usenet newsgroup post belonging to one of
20 topic categories (e.g. sci.space, talk.politics.guns, rec.autos).
"""

import os
import tarfile
import zipfile
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_zip(zip_path: str, output_dir: str) -> None:
    """Extract a ZIP archive to output_dir."""
    logger.info(f"Extracting ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    logger.info(f"ZIP extracted to: {output_dir}")


def extract_tar_gz(tar_path: str, output_dir: str) -> None:
    """Extract a TAR.GZ archive to output_dir."""
    logger.info(f"Extracting TAR.GZ: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(output_dir)
    logger.info(f"TAR.GZ extracted to: {output_dir}")


def load_newsgroups_documents(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Walk the extracted 20 Newsgroups directory tree and load all plain text
    documents.

    Returns
    -------
    texts  : list of document body strings
    labels : list of newsgroup category names (folder names)
    ids    : list of document identifiers (relative file paths)
    """
    texts, labels, ids = [], [], []
    data_path = Path(data_dir)

    categories = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not categories:
        raise FileNotFoundError(
            f"No newsgroup category folders found in: {data_dir}\n"
            "Make sure the TAR.GZ archive has been extracted first."
        )

    logger.info(f"Found {len(categories)} newsgroup categories.")

    for cat_dir in categories:
        category = cat_dir.name
        doc_files = sorted([f for f in cat_dir.iterdir() if f.is_file()])
        for doc_file in doc_files:
            try:
                raw = doc_file.read_bytes()
                text = raw.decode("utf-8", errors="replace")
                # Strip email headers (everything before first blank line)
                body_start = text.find("\n\n")
                body = text[body_start + 2:].strip() if body_start != -1 else text.strip()
                if len(body) > 50:          # discard near-empty documents
                    texts.append(body)
                    labels.append(category)
                    ids.append(str(doc_file.relative_to(data_path)))
            except Exception as exc:
                logger.warning(f"Could not read {doc_file}: {exc}")

    logger.info(f"Loaded {len(texts)} documents across {len(categories)} categories.")
    return texts, labels, ids


def prepare_dataset(
    zip_path: str,
    extract_root: str = "./data/raw",
    tar_name: str = "20_newsgroups.tar.gz",
) -> Tuple[List[str], List[str], List[str]]:
    """
    End-to-end pipeline:
      1. Extract ZIP → find TAR.GZ → extract TAR.GZ → load text documents.

    Parameters
    ----------
    zip_path     : path to the downloaded ZIP file
    extract_root : directory where archives are unpacked
    tar_name     : name of the TAR.GZ file inside the ZIP

    Returns
    -------
    texts, labels, ids
    """
    os.makedirs(extract_root, exist_ok=True)

    # Step 1: extract ZIP
    extract_zip(zip_path, extract_root)

    # Step 2: locate and extract TAR.GZ
    tar_path = os.path.join(extract_root, tar_name)
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Expected TAR.GZ not found at: {tar_path}")
    extract_tar_gz(tar_path, extract_root)

    # Step 3: load documents from extracted folder
    data_dir = os.path.join(extract_root, "20_newsgroups")
    texts, labels, ids = load_newsgroups_documents(data_dir)
    return texts, labels, ids
