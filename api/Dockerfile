FROM python:3.11-slim

WORKDIR /app

#sys dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

#py dependencies
RUN pip install --no-cache-dir -r requirements.txt

#application code
COPY app.py .

EXPOSE 8000

#health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]