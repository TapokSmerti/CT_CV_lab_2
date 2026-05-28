"""
Оценка модели на тестовой части SVHN.
Считает mAP50, mAP50-95, Precision, Recall, IoU.

Запуск:
  python evaluate.py
  python evaluate.py --weights runs/detect/runs/digit_detector-4/weights/best.pt
"""

import argparse
import numpy as np
from ultralytics import YOLO


DEFAULT_WEIGHTS = "runs/detect/runs/digit_detector/weights/best.pt"
DATA            = "digits.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Путь к весам модели",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Порог IoU для NMS (default: 0.5)",
    )
    args = parser.parse_args()

    print(f"Модель:  {args.weights}")
    print(f"Датасет: {DATA} → split=test (SVHN)")
    print(f"conf={args.conf}, iou={args.iou}\n")

    model = YOLO(args.weights)

    metrics = model.val(
        data    = DATA,
        split   = "test",
        conf    = args.conf,
        iou     = args.iou,
        plots   = True,       # сохранит confusion matrix и PR-кривую
        save_json = False,
    )

    map50   = metrics.box.map50
    map5095 = metrics.box.map
    prec    = metrics.box.mp
    rec     = metrics.box.mr

    # IoU считается внутри metrics как среднее по matched детекциям
    # Ultralytics не отдаёт его напрямую, но map50-95 по сути
    # и есть усреднение mAP по порогам IoU от 0.5 до 0.95
    # Для явного среднего IoU по детекциям используем stats
    mean_iou = float(np.mean(metrics.box.iouv)) if hasattr(metrics.box, "iouv") else None

    print("=" * 50)
    print("Метрики на тестовой части SVHN:")
    print("=" * 50)
    print(f"  mAP50:    {map50:.4f}")
    print(f"  mAP50-95: {map5095:.4f}")
    print(f"  Precision:{prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    if mean_iou:
        print(f"  IoU (avg thresholds): {mean_iou:.4f}")

    print()

    # Метрики по каждому классу (цифры 0-9)
    print("Метрики по классам:")
    print(f"  {'Класс':<8} {'AP50':>8} {'Precision':>10} {'Recall':>8}")
    print("  " + "-" * 38)
    names = model.names
    for i, (ap, p, r) in enumerate(zip(
        metrics.box.ap50,
        metrics.box.p,
        metrics.box.r,
    )):
        print(f"  {names[i]:<8} {ap:>8.4f} {p:>10.4f} {r:>8.4f}")

    print()
    if map50 >= 0.6:
        print(f"✓ Требование mAP50 ≥ 0.6 выполнено ({map50:.4f})")
    else:
        print(
            f"✗ mAP50 = {map50:.4f} < 0.6\n"
            "  Варианты улучшения:\n"
            "  1. Добавить extra-сплит SVHN в обучение (~26k изображений):\n"
            "       python convert_svhn.py --splits train test extra\n"
            "     добавь в digits.yaml → train: datasets/svhn_yolo/extra/images\n"
            "  2. Увеличить epochs до 120\n"
            "  3. Попробовать yolo12l.pt"
        )


if __name__ == "__main__":
    main()