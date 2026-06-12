# String-Art AI Generator — рабочий MVP

Локальный проект превращает MNIST в последовательности гвоздей, обучает
небольшой causal Transformer и генерирует новые continuous string-art пути
для цифр 0–9.

## Что исправлено относительно первоначального варианта

- Метки цифр имеют отдельные token ID и не пересекаются с гвоздями `0..255`.
- Добавлены `EOS` и `PAD`, поэтому последовательности могут иметь разную длину.
- Все линии Брезенхема вычисляются один раз, а не заново для каждого кандидата.
- Скоринг ослабляет смещение в пользу длинных хорд.
- Запрещены слишком короткие переходы, мгновенный возврат `A -> B -> A`
  и недавние повторения ребра.
- Датасет хранится в сжатом `.npz`, а не медленном текстовом файле.
- Есть validation split, AMP на CUDA, gradient clipping и best checkpoint.
- Генерация использует temperature/top-k/repetition penalty, а не только argmax.
- Рендер действительно может быть `4096 × 4096`: `8 inches × 512 DPI`.
- Линии рендерятся пакетно через `LineCollection`.

## Установка

Рекомендуется Python 3.11 или 3.12.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Для NVIDIA лучше поставить сборку PyTorch, подходящую вашей версии CUDA,
по команде из официального установщика PyTorch, а затем остальные зависимости.

## Быстрый запуск

### 1. Подготовить MVP-датасет

```bash
python prepare_dataset.py --samples 1000 --max-lines 180
```

Сначала разумно проверить небольшой прогон:

```bash
python prepare_dataset.py --samples 100 --max-lines 120 --output smoke_dataset.npz
python preview_dataset.py --dataset smoke_dataset.npz
```

Для более качественного эксперимента:

```bash
python prepare_dataset.py --samples 5000 --max-lines 220
```

Подготовка датасета обычно является самым медленным этапом. Точное время
сильно зависит от CPU, числа изображений, гвоздей и линий.

### 2. Обучить Transformer

```bash
python train_ai.py --epochs 15 --batch-size 32
```

На слабой видеокарте:

```bash
python train_ai.py --epochs 15 --batch-size 8
```

На CPU можно уменьшить модель:

```bash
python train_ai.py ^
  --epochs 10 ^
  --batch-size 16 ^
  --embed-dim 128 ^
  --heads 4 ^
  --layers 3 ^
  --ff-dim 384
```

```bash
python train_ai.py --epochs 10 --batch-size 16 --embed-dim 128 --heads 4 --layers 3 --ff-dim 384
```

В Linux/macOS замените `^` на `\`.

### 3. Сгенерировать цифру

```bash
python generate.py 7
```

Будут созданы:

- `digit_7_4k.png` — изображение 4096×4096;
- `digit_7_4k.json` — точная последовательность гвоздей.

Другие варианты:

```bash
python generate.py 3 --temperature 0.70 --top-k 16 --seed 10
python generate.py 9 --temperature 1.00 --top-k 48 --seed 77
python generate.py 4 --temperature 0 --top-k 1
```

`temperature 0` включает greedy/argmax-режим. Он детерминирован, но чаще
зацикливается и обычно выглядит беднее sampling-варианта.

## Структура

```text
string_art_ai_mvp/
├── geometry.py
├── model.py
├── prepare_dataset.py
├── preview_dataset.py
├── train_ai.py
├── generate.py
├── requirements.txt
└── README.md
```

## Честное позиционирование проекта

Это не text-to-image модель и не универсальный генератор картинок.
Модель изучает распределение последовательностей гвоздей для десяти классов
MNIST. Для GitHub/Habr лучше называть её:

> Class-conditioned autoregressive string-art generator.

Самая сильная демонстрация: рядом показать исходный MNIST, результат
детерминированного кодировщика и несколько разных AI-сэмплов одной цифры.

## Следующий уровень

После MVP наиболее полезны:

1. conditioning по embedding исходного изображения вместо одной цифры;
2. обучение на EMNIST для букв;
3. отдельный critic/raster loss, оценивающий реальный итоговый рисунок;
4. beam search с геометрическим штрафом;
5. экспорт последовательности в SVG и инструкции для физического станка.
