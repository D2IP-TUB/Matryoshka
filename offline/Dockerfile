FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc g++ make wget curl git \
    libpq-dev libopenblas-dev liblapack-dev \
    procps htop iotop \
    openssh-client \
    gzip unzip tar \
    bc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

RUN mkdir -p /app/data /app/logs /app/config /app/augmentation/utils/logging/logs

COPY ./offline/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./offline/ /app
COPY ./augmentation/utils /app/augmentation/utils
COPY ./augmentation/index.py /app/augmentation/index.py

RUN chown -R appuser:appuser /app \
    && chmod +x /app/run.sh

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f "python.*main.py" > /dev/null || exit 1

CMD ["tail", "-f", "/dev/null"]