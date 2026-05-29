from ultralytics import YOLO
from pathlib import Path
import cv2


MODEL_PATH = "runs/detect/runs/digit_detector-4/weights/best.pt"
SOURCE_DIR = "photos"
OUTPUT_DIR = "results"

CONFIDENCE = 0.5


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=SOURCE_DIR,
        conf=CONFIDENCE,
        save=True,
        project=OUTPUT_DIR,
        name="predictions",
        imgsz=640,
    )

    print(f"Processed {len(results)} images")

    for result in results:
        print(result.path)

        boxes = result.boxes

        for box in boxes:
            conf = float(box.conf[0])

            xyxy = box.xyxy[0].tolist()

            print(f"Detection: {xyxy}, conf={conf:.3f}")

    print("Inference complete!")


if __name__ == "__main__":
    main()
