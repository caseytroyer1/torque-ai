FROM python:3.11-bullseye

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    libgles2-mesa \
    mesa-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV EGL_PLATFORM=surfaceless

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

CMD ["gunicorn", "--chdir", "backend", "--timeout", "120", "--workers", "1", "app:app"]
