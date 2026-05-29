# Детектор цифр и номеров на уличных фотографиях

Лабораторная работа по компьютерному зрению. Детектирует отдельные цифры (0–9) на изображениях и группирует их в номера. Обучена на датасетах SVHN и NumberDetection, архитектура YOLOv12m.

---

## Структура проекта

```
CT_CV_lab_2/
├── datasets/
│   ├── svhn/                  # SVHN Format 1 (скачать вручную)
│   │   ├── train/             # 73257 изображений + digitStruct.mat
│   │   ├── test/              # 26032 изображений + digitStruct.mat
│   │   └── extra/             # 531131 изображений (опционально)
│   ├── svhn_yolo/             # SVHN в формате YOLO (генерируется скриптом)
│   └── number_detection/      # NumberDetection с Roboflow (скачать вручную)
├── photos/                    # Ваши уличные фотографии
├── results/                   # Результаты инференса
├── runs/                      # Артефакты обучения (веса, графики, метрики)
├── convert_svhn.py            # Конвертер SVHN → YOLO формат
├── digits.yaml                # Конфигурация датасета для обучения
├── train.py                   # Обучение модели
├── evaluate.py                # Оценка на тестовой части SVHN
├── inference.py               # Инференс на уличных фотографиях
└── requirements.txt
```

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone <repo_url>
cd CT_CV_lab_2
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Подготовка датасетов

### SVHN (Street View House Numbers)

1. Скачать **Format 1** с официального сайта: http://ufldl.stanford.edu/housenumbers/
   - `train.tar.gz` (~404 MB)
   - `test.tar.gz` (~276 MB)

2. Распаковать в `datasets/svhn/`:

```bash
mkdir -p datasets/svhn
tar -xzf train.tar.gz -C datasets/svhn/
tar -xzf test.tar.gz  -C datasets/svhn/
```

   Ожидаемая структура:
   ```
   datasets/svhn/
   ├── train/
   │   ├── 1.png, 2.png, ...
   │   └── digitStruct.mat
   └── test/
       ├── 1.png, 2.png, ...
       └── digitStruct.mat
   ```

3. Сконвертировать в формат YOLO:

```bash
python convert_svhn.py --svhn_root datasets/svhn --output_root datasets/svhn_yolo
```

   Скрипт создаст `datasets/svhn_yolo/` с папками `train/images/`, `train/labels/`, `test/images/`, `test/labels/`.

   > **Опционально:** если хочется больше обучающих данных (extra содержит ~531k изображений):
   > ```bash
   > tar -xzf extra.tar.gz -C datasets/svhn/
   > python convert_svhn.py --svhn_root datasets/svhn --output_root datasets/svhn_yolo --splits train test extra
   > ```
   > Затем добавить в `digits.yaml` в секцию `train`:
   > ```yaml
   > - datasets/svhn_yolo/extra/images
   > ```

### NumberDetection

1. Скачать датасет с Roboflow вручную:
   https://universe.roboflow.com/university-of-toronto-xho85/numberdetection-eppfj/dataset/2
   - Выбрать формат **YOLOv8**
   - Распаковать в `datasets/number_detection/`

   Ожидаемая структура:
   ```
   datasets/number_detection/
   ├── train/images/, train/labels/
   ├── valid/images/, valid/labels/
   ├── test/images/,  test/labels/
   └── data.yaml
   ```

---

## Обучение

```bash
python train.py
```

Модель обучается на объединённом датасете (NumberDetection train + SVHN train). Веса сохраняются в `runs/detect/runs/digit_detector/weights/`.

**Параметры обучения** (можно изменить в `train.py`):

| Параметр | Значение | Описание |
|----------|----------|----------|
| `MODEL`  | `yolo12m.pt` | Базовая модель, скачается автоматически |
| `epochs` | 80 | Максимум эпох (early stopping при patience=20) |
| `batch`  | 32 | Размер батча — подбери под свою GPU |
| `imgsz`  | 640 | Размер входного изображения |
| `device` | 0 | GPU индекс; `"cpu"` если нет GPU |

> **Подбор batch под GPU:**
> - 8 GB VRAM → batch=16
> - 16 GB VRAM → batch=32
> - 24+ GB VRAM → batch=64
>
> Ultralytics автоматически снизит batch если не хватает памяти.

**Мониторинг обучения** — в соседнем терминале:

```bash
watch -n2 nvidia-smi
```

После обучения в папке `runs/detect/runs/digit_detector/` будут:
- `weights/best.pt` — лучшие веса по val метрике
- `weights/last.pt` — веса последней эпохи
- `results.csv` — метрики по эпохам
- `results.png` — графики loss и метрик

---

## Оценка качества на тестовом датасете SVHN

```bash
python evaluate.py
```

По умолчанию берёт веса из `runs/detect/runs/digit_detector/weights/best.pt`. Чтобы указать другие:

```bash
python evaluate.py --weights runs/detect/runs/digit_detector-4/weights/best.pt
```

Дополнительные параметры:

```bash
python evaluate.py --conf 0.25 --iou 0.5
```

**Пример вывода:**

```
Метрики на тестовой части SVHN:
==================================================
  mAP50:    0.7231
  mAP50-95: 0.4812
  Precision:0.8104
  Recall:   0.6943

Метрики по классам:
  Класс       AP50  Precision   Recall
  ----------------------------------------
  0           0.71      0.80     0.68
  1           0.75      0.83     0.71
  ...

✓ Требование mAP50 ≥ 0.6 выполнено (0.7231)
```

> Требование по заданию: **mAP50 ≥ 0.6** на тестовой части SVHN.
>
> Если результат ниже 0.6 — скрипт сам подскажет что делать.

---

## Инференс на уличных фотографиях

Положите свои фотографии в папку `photos/` и запустите:

```bash
python inference.py
```

Или явно указать путь:

```bash
python inference.py --source photos/
python inference.py --source photos/my_photo.jpg   # одно фото
```

**Как работает группировка цифр в номера:**

Модель детектирует каждую цифру отдельно. Затем скрипт объединяет соседние детекции в номера: боксы на одной горизонтальной строке с небольшим промежутком между ними считаются одним номером и читаются слева направо.

**Пример вывода в консоли:**

```
  photo_01.jpg: 6 цифр → 2 номер(ов)
    «1234» bbox=[102, 45, 310, 98] conf=0.87
    «56»   bbox=[400, 210, 480, 250] conf=0.72
```

**Результаты** сохраняются в `results/predictions/` — те же фото с нарисованными боксами:
- серые тонкие боксы — отдельные цифры с уверенностью
- зелёные толстые боксы — номера целиком с прочитанным текстом

Настройка порогов (если модель находит слишком много/мало):

```bash
python inference.py --conf 0.4 --iou 0.45
```

---

## Если mAP < 0.6

По убыванию эффекта:

1. **Добавить extra-сплит SVHN** — ещё 531k изображений, самый сильный буст:
   ```bash
   tar -xzf extra.tar.gz -C datasets/svhn/
   python convert_svhn.py --splits train test extra
   # добавить в digits.yaml → train: - datasets/svhn_yolo/extra/images
   python train.py
   ```

2. **Увеличить epochs** — в `train.py` поставить `epochs=120`

3. **Взять модель побольше** — в `train.py` заменить `MODEL = "yolo12l.pt"`

---

## Зависимости

| Библиотека | Назначение |
|------------|-----------|
| `ultralytics` | YOLOv12, обучение и инференс |
| `torch`, `torchvision` | PyTorch |
| `opencv-python` | Чтение и визуализация изображений |
| `h5py` | Чтение `.mat` файлов SVHN |
| `numpy` | Работа с массивами |
| `roboflow` | (опционально) автоматическая загрузка датасетов |
