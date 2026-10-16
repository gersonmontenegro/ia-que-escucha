FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rag ./rag
COPY tests ./tests
COPY app.py .

EXPOSE 7860
CMD ["python", "app.py"]
