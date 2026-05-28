from pathlib import Path
from roboflow import Roboflow


BASE_DIR = Path("datasets")

SVHN_DIR = BASE_DIR / "svhn"
NUMBER_DIR = BASE_DIR / "number_detection"

SVHN_DIR.mkdir(parents=True, exist_ok=True)
NUMBER_DIR.mkdir(parents=True, exist_ok=True)


ROBOFLOW_API_KEY = "lPpWX1mnE2k3R4I1YpAR"


def download_svhn():
    print("Downloading SVHN YOLO dataset from Roboflow...")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    project = rf.workspace("ufldl-stanford").project("svhn")

    dataset = project.version(1).download(
        "yolov8",
        location=str(SVHN_DIR)
    )

    print("SVHN downloaded!")


def download_number_detection():
    print("Downloading NumberDetection dataset...")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    project = rf.workspace("university-of-toronto-xho85").project(
        "numberdetection-eppfj"
    )

    dataset = project.version(2).download(
        "yolov8",
        location=str(NUMBER_DIR)
    )

    print("NumberDetection downloaded!")


if __name__ == "__main__":
    download_svhn()
    download_number_detection()

    print("All datasets downloaded successfully!")
