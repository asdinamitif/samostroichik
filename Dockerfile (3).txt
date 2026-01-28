FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Для некоторых зависимостей нужен компилятор
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
  && rm -rf /var/lib/apt/lists/*

# Ставим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота
COPY . .

# Команда по умолчанию (Railway все равно переопределит startCommand из railway.json)
CMD ["python", "bot.py"]
