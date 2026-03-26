import os
import urllib.request
import zipfile

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")

def fetch_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if os.path.exists(os.path.join(DATA_DIR, "SMSSpamCollection")):
        print("Dataset already exists. Skipping download.")
        return

    print(f"Downloading dataset from {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        print("Extraction complete. Dataset is ready in the 'data' folder.")
    except Exception as e:
        print(f"Error fetching dataset: {e}")

if __name__ == "__main__":
    fetch_dataset()
