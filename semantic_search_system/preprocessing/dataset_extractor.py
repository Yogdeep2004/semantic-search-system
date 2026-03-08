import zipfile
import tarfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DATASET_ZIP, DATASET_TAR, EXTRACTED_DATA_DIR, EXTRACTED_FOLDER

def extract_dataset():
    """
    Extracts the twenty_newsgroup_dataset.zip and then the 20_newsgroups.tar.gz.
    Idempotent: skips extraction if files already exist.
    """
    if not DATASET_ZIP.exists():
        print(f"Error: Dataset ZIP not found at {DATASET_ZIP}")
        return

    # 1. Extract ZIP
    if not DATASET_TAR.exists():
        print(f"Extracting {DATASET_ZIP} to {EXTRACTED_DATA_DIR}...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_DATA_DIR)
        print("ZIP extraction complete.")
    else:
        print(f"{DATASET_TAR} already exists. Skipping ZIP extraction.")

    # 2. Extract TAR.GZ
    if not EXTRACTED_FOLDER.exists():
        if DATASET_TAR.exists():
            print(f"Extracting {DATASET_TAR} to {EXTRACTED_DATA_DIR}...")
            with tarfile.open(DATASET_TAR, "r:gz") as tar_ref:
                tar_ref.extractall(EXTRACTED_DATA_DIR)
            print("TAR extraction complete.")
        else:
            # Check if it was extracted into a subfolder by ZipFile or if we need to search for it
            # The user said the ZIP contains 20_newsgroups.tar.gz
            # Let's check if it's there after ZIP extraction
            print(f"Searching for {DATASET_TAR.name} in {EXTRACTED_DATA_DIR}...")
            # Re-check existence as it might have been extracted
            if DATASET_TAR.exists():
                with tarfile.open(DATASET_TAR, "r:gz") as tar_ref:
                    tar_ref.extractall(EXTRACTED_DATA_DIR)
                print("TAR extraction complete.")
            else:
                print(f"Error: {DATASET_TAR.name} not found after ZIP extraction.")
    else:
        print(f"{EXTRACTED_FOLDER} already exists. Skipping TAR extraction.")

if __name__ == "__main__":
    extract_dataset()
