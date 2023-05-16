# syntax = docker/dockerfile:1.0-experimental

FROM python:3.9.16

COPY ./demo /app

WORKDIR /app

RUN --mount=type=secret,id=my_env,dst=/etc/.env \
    pip install -r requirements.txt

EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port 7860