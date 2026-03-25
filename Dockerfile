FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libglx-mesa0 \
    libegl1 \
    libegl-mesa0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

CMD ["gunicorn", "--chdir", "backend", "--timeout", "120", "--workers", "1", "app:app"]
