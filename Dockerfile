# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости для PyTorch и transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения
COPY . .

# Устанавливаем переменную окружения для Python пути
# Это позволяет импортировать модули из lib
ENV PYTHONPATH=/app:$PYTHONPATH

# Открываем порт для FastAPI
EXPOSE 8000

# Команда запуска приложения
CMD ["uvicorn", "lib.main:app", "--host", "0.0.0.0", "--port", "8000"]

