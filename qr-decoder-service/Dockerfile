FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install \
       --no-install-recommends \
       -y libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install \
    --upgrade pip \
    && python -m pip install \
       -r requirements.txt

COPY app.py .

RUN python -c \
    "from pyzbar.pyzbar import decode; print('Pyzbar and ZBar loaded successfully')"

EXPOSE 8000

CMD [
  "uvicorn",
  "app:app",
  "--host",
  "0.0.0.0",
  "--port",
  "8000"
]
