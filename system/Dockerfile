FROM python:3.11-slim

WORKDIR /app

COPY system/requirements.txt /app/system/requirements.txt
RUN pip install --no-cache-dir -r /app/system/requirements.txt

COPY . /app

ENV API_PROFILE=full
EXPOSE 8000

CMD ["sh", "-c", "python system/service_api.py --db-path /data/market_intel.db --host 0.0.0.0 --port 8000 --profile ${API_PROFILE}"]
