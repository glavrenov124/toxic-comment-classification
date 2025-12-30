# Toxic Comment Classification (MLOps)

Проект решает задачу **multi-label классификации токсичных комментариев**: по тексту пользовательского комментария модель предсказывает вероятности и бинарные метки для 6 типов токсичности:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Источник: HuggingFace датасет  
`thesofakillers/jigsaw-toxic-comment-classification-challenge`
## Models
В проекте реализованы две модели: 
- **Baseline**: TF-IDF (1–2 word ngrams) + One-Vs-Rest Logistic Regression; 
- **TextCNN (основная)**: сверточная модель на **PyTorch Lightning**, выдаёт 6 логитов по числу классов.

## Metrics
Логируются основные метрики: **ROC-AUC (macro)**, **F1 (micro)**, **F1 (macro)**, а также **AUC по каждому классу** (когда возможно посчитать на сплите).

## Инструкция по установке
### 1. Клонирование репозитория
```bash
git clone https://github.com/glavrenov124/toxic-comment-classification
cd toxic-comment-classification
```

### 2. Установка зависимостей
```bash
poetry install
```
### 3. Установка хуков для pre-commit
```bash
poetry run pre-commit install
```
### 4. Запуск проверки всех файлов проекта
```bash
poetry run pre-commit run -a
```
### 5. Поднятие MLflow
```bash
poetry run mlflow server
```
UI будет доступен по адресу http://127.0.0.1:8080

## Data management (DVC)

Данные не хранятся в git — используется DVC. На старте обучения код пытается выполнить dvc pull для путей data/raw/jigsaw и data/processed. Если DVC remote доступен, можно подтянуть данные вручную:
```bash
poetry run dvc pull
```
Если DVC remote недоступен (например, локальный remote на другой машине), пайплайн автоматически скачает датасет из HuggingFace (fallback download_data()), поэтому обучение всё равно должно запускаться.
## Train

### Тренировка baseline
```bash
poetry run python -m toxic_comment_classification.commands command=baseline
```
### Тренировка основной модели - textcnn
```bash
poetry run python -m toxic_comment_classification.commands command=textcnn
```










