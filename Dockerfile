FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY recommendation/ ./recommendation/
COPY models/ ./models/

# Set Python path so 'recommendation' becomes importable
ENV PYTHONPATH=/app

CMD ["python", "recommendation/main.py"]

COPY raw_data/ ./raw_data/
