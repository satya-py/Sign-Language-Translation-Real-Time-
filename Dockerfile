FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY isl-translator/backend/requirements.txt /tmp/req-ml.txt
COPY signai-extension/server/requirements.txt /tmp/req-flask.txt
RUN pip install --no-cache-dir -r /tmp/req-ml.txt -r /tmp/req-flask.txt

COPY isl-translator/live/ isl-translator/live/
COPY signai-extension/server/ signai-extension/server/

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app/signai-extension/server
EXPOSE 8080
CMD ["sh", "-c", "python app.py --host 0.0.0.0 --port ${PORT:-8080}"]