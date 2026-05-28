import os
import urllib.request
from pathlib import Path

from roboflow import Roboflow


BASE_DIR = Path("datasets")
SVHN_DIR = BASE_DIR / "svhn"
NUMBER_DIR = BASE_DIR / "number_detection"

SVHN_DIR.mkdir(parents=True, exist_ok=True)
NUMBER_DIR.mkdir(parents=True, exist_ok=True)


def download_svhn():
    print("Downloading SVHN dataset...")

    urls = {
        "train": "http://ufldl.stanford.edu/housenumbers/train.tar.gz",
        "test": "http://ufldl.stanford.edu/housenumbers/test.tar.gz",
        "extra": "http://ufldl.stanford.edu/housenumbers/extra.tar.gz",
    }

    for split, url in urls.items():
        output_path = SVHN_DIR / f"{split}.tar.gz"

        if output_path.exists():
            print(f"{split} already downloaded")
            continue

        print(f"Downloading {split}...")
        urllib.request.urlretrieve(url, output_path)

    print("SVHN downloaded!")


def download_roboflow_dataset():
    print("Downloading NumberDetection dataset from Roboflow...")

    ROBOFLOW_API_KEY = "lPpWX1mnE2k3R4I1YpAR"

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    project = rf.workspace("university-of-toronto-xho85").project(
        "numberdetection-eppfj"
    )

    dataset = project.version(2).download("yolov8", location=str(NUMBER_DIR))

    print("Roboflow dataset downloaded!")


if __name__ == "__main__":
    download_svhn()
    download_roboflow_dataset()

    print("All datasets downloaded successfully!")