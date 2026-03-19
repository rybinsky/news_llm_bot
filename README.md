# 🤖 AI-агент: Генератор мемов по последним новостям

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-2.0+-purple.svg)](https://python-poetry.org/)

## Описание

AI-агент, который автоматически собирает новости с популярных источников, классифицирует их по темам и генерирует остроумные шутки и мемы на основе актуальных событий.

**Основные возможности:**
- 🗞️ Автоматический сбор новостей с Lenta.ru и RIA.ru
- 🏷️ Классификация новостей по темам (Политика, Наука, Спорт, Технологии, Культура)
- 🎭 Генерация шуток и мемов на основе RAG (Retrieval-Augmented Generation)
- 🔍 Семантический поиск релевантных новостей
- 🖥️ Веб-интерфейс на Streamlit

## Технологии

**ML & NLP:**
- LangChain, Ollama (LLaMA 3.1)
- Hugging Face Transformers (rubert-tiny2)
- FAISS (векторный поиск)

**Backend & Database:**
- Python 3.10+, Streamlit
- PostgreSQL, SQLAlchemy
- Newspaper3k (парсинг новостей)

**Code Quality:**
- Black, isort, Flake8, mypy, bandit, ruff
- Pre-commit hooks

## Структура проекта

```
news_llm_bot/
├── src/bot/
│   ├── app.py              # Streamlit приложение
│   ├── parser.py           # Парсер новостей
│   ├── models/             # Модели данных
│   ├── services/           # Бизнес-логика
│   │   ├── classifier.py   # Классификатор тем
│   │   ├── database.py     # Работа с БД
│   │   ├── scraper.py      # Парсинг статей
│   │   └── ...
│   └── config/
│       └── config.yaml     # Конфигурация
├── tests/
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Установка и запуск

### 1. Установка зависимостей

```bash
# Установка Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей проекта
poetry install
```

### 2. Установка Ollama и модели

```bash
# Установка Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка модели LLaMA 3.1
ollama pull llama3.1:8b
ollama serve  # запуск сервера
```

### 3. Настройка базы данных

```bash
# Создание базы данных PostgreSQL
createdb news_db
```

### 4. Настройка переменных окружения

Создайте файл `.env` в корневой директории:

```bash
POSTGRES_USER=your_username
POSTGRES_PASS=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=news_db
```

### 5. Запуск

**Парсинг новостей:**
```bash
poetry run python src/bot/parser.py
```

**Веб-приложение:**
```bash
poetry run streamlit run src/bot/app.py
```

Приложение будет доступно по адресу: http://localhost:8501

## Конфигурация

Основные настройки в `src/config/config.yaml`:

```yaml
ollama:
  model: "llama3.1:8b"

classifier:
  topics: ["Политика", "Наука", "Спорт", "Технологии", "Культура", "Разное"]
  temperature: 0.0

generator:
  model: "llama3.1:8b"
  temperature: 2.0

text_embedder: cointegrated/rubert-tiny2

news_sources:
  lenta:
    url: https://lenta.ru
  ria:
    url: https://ria.ru
```

## Разработка

```bash
# Установка pre-commit hooks
poetry run pre-commit install

# Запуск проверок
poetry run pre-commit run --all-files

# Запуск тестов
poetry run pytest
```

## Авторы

Rybinsky (barinov.na@phystech.edu)
