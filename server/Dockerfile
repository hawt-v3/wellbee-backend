FROM python:3.8

RUN pip install Flask gunicorn tensorflow openai numpy pandas nltk

COPY src/ app/
WORKDIR /app

ENV PORT=8080

CMD exec gunicorn --bind :$PORT --workers 3 --threads 8 --timeout 0 app:app


