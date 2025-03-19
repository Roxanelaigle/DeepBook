FROM python:3.10.6-buster

RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY .cache .cache

COPY API_package API_package
COPY covers_to_text covers_to_text
COPY gbooks_api gbooks_api
COPY main_pipeline main_pipeline
COPY models models
COPY raw_data raw_data
COPY recommendation recommendation

COPY requirements.txt requirements.txt
COPY setup.py setup.py

ENV HF_HOME=/.cache

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn API_package.API_package:app --host 0.0.0.0 --port $PORT
