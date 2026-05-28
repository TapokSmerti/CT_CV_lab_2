"""
Конвертер SVHN Format 1 → YOLO

Структура SVHN Format 1:
  train/
    1.png, 2.png, ...
    digitStruct.mat   ← bbox + labels для каждого изображения
  test/
    ...
  extra/
    ...

Запуск:
  python convert_svhn.py --svhn_root datasets/svhn --output_root datasets/svhn_yolo
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import shutil


def get_box_data(index, hdf5_data):
    """
    Извлекает данные одного поля (top/left/width/height/label)
    для изображения с заданным индексом.
    Возвращает list значений (один элемент или несколько цифр).
    """
    ref = hdf5_data["/digitStruct/bbox"][index, 0]
    bbox = hdf5_data[ref]

    result = {}
    for key in ("top", "left", "width", "height", "label"):
        raw = bbox[key]
        if raw.shape[0] == 1:
            # одна цифра — значение хранится напрямую
            result[key] = [int(raw[0, 0])]
        else:
            # несколько цифр — каждый элемент это ссылка на объект
            vals = []
            for i in range(raw.shape[0]):
                ref2 = raw[i, 0]
                val = int(hdf5_data[ref2][0, 0])
                vals.append(val)
            result[key] = vals

    return result


def get_img_name(index, hdf5_data):
    """Возвращает имя файла изображения по индексу."""
    ref = hdf5_data["/digitStruct/name"][index, 0]
    name_arr = hdf5_data[ref]
    # символы хранятся как uint16
    return "".join(chr(c) for c in name_arr[:, 0])


def convert_split(svhn_split_dir: Path, output_split_dir: Path):
    """
    Конвертирует один сплит (train/test/extra).
    Создаёт:
      output_split_dir/images/  — копии PNG
      output_split_dir/labels/  — .txt с YOLO-разметкой
    """
    mat_path = svhn_split_dir / "digitStruct.mat"
    if not mat_path.exists():
        print(f"  [SKIP] {mat_path} не найден")
        return

    images_out = output_split_dir / "images"
    labels_out = output_split_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    skipped = 0
    converted = 0

    with h5py.File(mat_path, "r") as f:
        n = f["/digitStruct/name"].shape[0]
        print(f"  Всего изображений: {n}")

        for idx in range(n):
            if idx % 2000 == 0:
                print(f"  ... {idx}/{n}")

            img_name = get_img_name(idx, f)
            src_img = svhn_split_dir / img_name

            if not src_img.exists():
                skipped += 1
                continue

            # Размер изображения нужен для нормализации координат
            with Image.open(src_img) as img:
                W, H = img.size

            box_data = get_box_data(idx, f)
            n_digits = len(box_data["label"])

            lines = []
            valid = True

            for i in range(n_digits):
                label = box_data["label"][i]
                # В SVHN метка 10 означает цифру 0
                if label == 10:
                    label = 0

                top    = box_data["top"][i]
                left   = box_data["left"][i]
                width  = box_data["width"][i]
                height = box_data["height"][i]

                # Клипаем выход за границы
                left   = max(0, left)
                top    = max(0, top)
                right  = min(W, left + width)
                bottom = min(H, top + height)

                w = right - left
                h = bottom - top

                if w <= 0 or h <= 0:
                    valid = False
                    break

                # YOLO формат: class cx cy w h  (нормализованные 0..1)
                cx = (left + w / 2) / W
                cy = (top  + h / 2) / H
                nw = w / W
                nh = h / H

                # class_id: используем метку цифры (0-9)
                lines.append(f"{label} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if not valid or not lines:
                skipped += 1
                continue

            # Копируем изображение
            dst_img = images_out / img_name
            shutil.copy2(src_img, dst_img)

            # Сохраняем разметку
            stem = Path(img_name).stem
            label_file = labels_out / f"{stem}.txt"
            label_file.write_text("\n".join(lines))

            converted += 1

    print(f"  Готово: {converted} изображений, пропущено: {skipped}")


def write_yaml(output_root: Path):
    """Пишет svhn_yolo.yaml для Ultralytics."""
    yaml_content = f"""path: {output_root.resolve()}

train: train/images
val: test/images

# Классы: цифры 0-9
nc: 10
names:
  0: "0"
  1: "1"
  2: "2"
  3: "3"
  4: "4"
  5: "5"
  6: "6"
  7: "7"
  8: "8"
  9: "9"
"""
    yaml_path = output_root / "svhn_yolo.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nYAML сохранён: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Конвертер SVHN Format 1 → YOLO")
    parser.add_argument(
        "--svhn_root",
        type=str,
        default="datasets/svhn",
        help="Папка с распакованным SVHN (содержит train/, test/, extra/)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="datasets/svhn_yolo",
        help="Куда сохранить датасет в формате YOLO",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Какие сплиты конвертировать (train test extra)",
    )
    args = parser.parse_args()

    svhn_root   = Path(args.svhn_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n=== Конвертируем {split} ===")
        convert_split(svhn_root / split, output_root / split)

    write_yaml(output_root)
    print("\nКонвертация завершена!")
    print(f"Датасет: {output_root}")
    print("Используй svhn_yolo.yaml в train.py")


if __name__ == "__main__":
    main()