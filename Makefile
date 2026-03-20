PYTHON ?= python
DB_PATH ?= ./system/market_intel.db
HOST ?= 0.0.0.0
PORT ?= 8000
PROFILE ?= full

.PHONY: ingest api api-market-data api-agent-runtime api-profiles

ingest:
	$(PYTHON) system/daily_pipeline.py --repo-root . --db-path $(DB_PATH)

api:
	$(PYTHON) system/service_api.py --db-path $(DB_PATH) --host $(HOST) --port $(PORT) --profile $(PROFILE)

api-market-data:
	$(MAKE) api PROFILE=market-data

api-agent-runtime:
	$(MAKE) api PROFILE=agent-runtime

api-profiles:
	$(PYTHON) system/service_api.py --list-profiles
