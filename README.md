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

##Инструкция по установке
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

## Train

### Тренировка baseline
```bash
poetry run python -m toxic_comment_classification.commands command=baseline
```
### Тренировка основной модели - textcnn
```bash
poetry run python -m toxic_comment_classification.commands command=textcnn
```



