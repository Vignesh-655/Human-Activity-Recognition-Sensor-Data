import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_PATH = RAW_DIR / "uci_har.zip"
EXTRACT_DIR = RAW_DIR / "UCI_HAR_Dataset"

def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not (EXTRACT_DIR / "UCI HAR Dataset").exists() and not (EXTRACT_DIR / "train").exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(DATASET_URL, headers=headers)
        with urllib.request.urlopen(req) as response, open(ZIP_PATH, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete. Extracting outer archive...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        
        # Check if it has an inner zip file: UCI HAR Dataset.zip
        inner_zip = RAW_DIR / "UCI HAR Dataset.zip"
        if inner_zip.exists():
            print("Extracting inner archive UCI HAR Dataset.zip...")
            with zipfile.ZipFile(inner_zip, 'r') as inner_ref:
                inner_ref.extractall(RAW_DIR)
        print("Dataset extracted successfully!")
    else:
        print("Dataset already downloaded and extracted.")

if __name__ == "__main__":
    download_and_extract()
