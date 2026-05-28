from ultralytics import YOLO


def main():
    MODEL_NAME = "yolo12s.pt"

    model = YOLO(MODEL_NAME)

    model.train(
        data="number.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=12,

        project="runs",
        name="number_detector",

        pretrained=True,

        optimizer="AdamW",
        lr0=1e-3,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0005,

        flipud=0.0,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        patience=20,

        save=True,
        plots=True,
    )

    metrics = model.val()

    print("\nValidation metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
