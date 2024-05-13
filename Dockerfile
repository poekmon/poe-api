FROM python:3.11-alpine AS base

RUN apk add --no-cache build-base

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM base AS runner

COPY . /app
WORKDIR /app

CMD ["sh", "-c", "python main.py"]
