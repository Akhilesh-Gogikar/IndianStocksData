PYTHON ?= python
DB_PATH ?= ./system/market_intel.db
TICKERTAPE_DB ?= ./local_repository/tickertape.sqlite
HOST ?= 0.0.0.0
PORT ?= 8000
PROFILE ?= full
WEBHOOK_URL ?=
WEBHOOK_LIMIT ?= 50
TICKERTAPE_COMPANY_LIST ?= ./full-company-list.json
TICKERTAPE_MIN_SUCCESS_RATE ?= 0.98
TICKERTAPE_SNAPSHOT_DATE ?=
TICKERTAPE_PIPELINE_ARGS = --db $(TICKERTAPE_DB) --company-list $(TICKERTAPE_COMPANY_LIST) --target-db $(DB_PATH) --min-success-rate $(TICKERTAPE_MIN_SUCCESS_RATE) $(if $(TICKERTAPE_SNAPSHOT_DATE),--snapshot-date $(TICKERTAPE_SNAPSHOT_DATE),)

.PHONY: ingest canonicalize api api-market-data api-agent-runtime api-profiles deliver-webhooks tickertape-daily tickertape-repair tickertape-gate tickertape-publish tickertape-status

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

tickertape-daily:
	bash scripts/run_tickertape_publish_pipeline.sh --mode daily $(TICKERTAPE_PIPELINE_ARGS)

tickertape-repair:
	bash scripts/run_tickertape_publish_pipeline.sh --mode repair $(TICKERTAPE_PIPELINE_ARGS)

tickertape-gate:
	$(PYTHON) tools/tickertape_publish_gate.py --db $(TICKERTAPE_DB) --company-list $(TICKERTAPE_COMPANY_LIST) --min-success-rate $(TICKERTAPE_MIN_SUCCESS_RATE) $(if $(TICKERTAPE_SNAPSHOT_DATE),--snapshot-date $(TICKERTAPE_SNAPSHOT_DATE),)

tickertape-publish:
	bash scripts/run_tickertape_publish_pipeline.sh --mode publish-only $(TICKERTAPE_PIPELINE_ARGS)

tickertape-status:
	$(PYTHON) tools/tickertape_status.py --db $(TICKERTAPE_DB)
