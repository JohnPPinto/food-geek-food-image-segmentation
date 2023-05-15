FROM python:3.9.16
COPY ./demo /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD uvicorn app:app --host 0.0.0.0 --port 7860