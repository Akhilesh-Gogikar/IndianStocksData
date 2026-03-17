FROM python:3.11-slim

WORKDIR /app

COPY system/requirements.txt /app/system/requirements.txt
RUN pip install --no-cache-dir -r /app/system/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python", "system/service_api.py", "--db-path", "/data/market_intel.db", "--host", "0.0.0.0", "--port", "8000"]
