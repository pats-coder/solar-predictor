FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --only-binary=:all: -r /app/requirements.txt
COPY app /app

CMD ["python", "-u", "/app/main.py"]
