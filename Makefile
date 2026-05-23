PYTHON ?= python
DB_PATH ?= ./system/market_intel.db
TICKERTAPE_DB ?= ./local_repository/tickertape.sqlite
HOST ?= 0.0.0.0
PORT ?= 8000
PROFILE ?= full
WEBHOOK_URL ?=
WEBHOOK_LIMIT ?= 50

.PHONY: ingest canonicalize api api-market-data api-agent-runtime api-profiles deliver-webhooks

ingest:
	$(PYTHON) system/daily_pipeline.py --repo-root . --db-path $(DB_PATH)

canonicalize:
	$(PYTHON) system/canonical_tickertape.py --source-db $(TICKERTAPE_DB) --target-db $(DB_PATH)

api:
	$(PYTHON) system/service_api.py --db-path $(DB_PATH) --host $(HOST) --port $(PORT) --profile $(PROFILE)

api-market-data:
	$(MAKE) api PROFILE=market-data

api-agent-runtime:
	$(MAKE) api PROFILE=agent-runtime

api-profiles:
	$(PYTHON) system/service_api.py --list-profiles

deliver-webhooks:
	$(PYTHON) system/webhook_worker.py --db-path $(DB_PATH) --limit $(WEBHOOK_LIMIT) $(if $(WEBHOOK_URL),--endpoint-url "$(WEBHOOK_URL)",)
