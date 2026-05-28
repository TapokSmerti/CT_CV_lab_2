"""
Обучение детектора цифр (0-9) на объединённом датасете
NumberDetection + SVHN → yolo12m

Запуск:
  python train.py

Финальные веса: runs/digit_detector/weights/best.pt
"""

from ultralytics import YOLO


DATA     = "digits.yaml"
MODEL    = "yolo12m.pt"
PROJECT  = "runs"
NAME     = "digit_detector"
DEVICE   = 0          # GPU; поменяй на "cpu" если нет GPU


def main():
    model = YOLO(MODEL)

    model.train(
        data      = DATA,
        epochs    = 80,
        imgsz     = 640,
        batch     = 16,
        device    = DEVICE,
        workers   = 8,

        project   = PROJECT,
        name      = NAME,

        pretrained  = True,
        optimizer   = "AdamW",
        lr0         = 1e-3,
        lrf         = 0.01,       # финальный lr = lr0 * lrf = 1e-5

        # Аугментации — SVHN уличные фото, нужна вариативность
        hsv_h       = 0.02,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 5.0,
        translate   = 0.1,
        scale       = 0.4,
        shear       = 2.0,
        perspective = 0.0005,
        flipud      = 0.0,        # цифры не переворачиваются вертикально
        fliplr      = 0.5,
        mosaic      = 1.0,
        mixup       = 0.1,

        patience  = 20,           # early stopping
        save      = True,
        plots     = True,
    )

    # Финальная оценка на test-части SVHN
    print("\n" + "="*50)
    print("Финальные метрики на тестовой части SVHN:")
    print("="*50)

    metrics = model.val(
        data   = DATA,
        split  = "test",
    )

    map50    = metrics.box.map50
    map5095  = metrics.box.map
    prec     = metrics.box.mp
    rec      = metrics.box.mr

    print(f"  mAP50:    {map50:.4f}")
    print(f"  mAP50-95: {map5095:.4f}")
    print(f"  Precision:{prec:.4f}")
    print(f"  Recall:   {rec:.4f}")

    if map50 >= 0.6:
        print(f"\n✓ Требование mAP50 ≥ 0.6 выполнено ({map50:.4f})")
    else:
        print(
            f"\n✗ mAP50 = {map50:.4f} — ниже 0.6.\n"
            "  Варианты:\n"
            "  1. Добавить extra-сплит SVHN в train (26000 доп. изображений):\n"
            "     в digits.yaml добавь в train: datasets/svhn_yolo/extra/images\n"
            "     и сконвертируй: python convert_svhn.py --splits train test extra\n"
            "  2. Увеличить epochs до 120\n"
            "  3. Попробовать yolo12l.pt"
        )


if __name__ == "__main__":
    main()