REGION=asia-northeast1
PROJECT=yukob-horse-racing
ACCOUNT=yukob.formal@gmail.com
GLOUD_WIDE_FLAG=--account=$(ACCOUNT) --project=$(PROJECT)

# Required:
#  - RACE_DATE: e.g. 20210101
.PHONY: execute_job_scrape_netkeiba
execute_job_scrape_netkeiba:
	gcloud run jobs execute scrape-netkeiba-job \
		--args="--race-date,$(RACE_DATE)" \
		--region=$(REGION) \
		$(GLOUD_WIDE_FLAG)

# Required:
#  - RACE_DATES: e.g. 20210101,20210102
.PHONY: execute_workflow_scrape_netkeiba
execute_workflow_scrape_netkeiba:
	gcloud workflows execute scrape-netkeiba \
		--location=$(REGION) \
		--data='{"race_dates": [$(RACE_DATES)]}' \
		$(GLOUD_WIDE_FLAG)

# Required:
#  - YEAR: e.g. 2021
#  - MONTH: e.g. 1
.PHONY: execute_job_scrape_netkeiba_race_dates
execute_job_scrape_netkeiba_race_dates:
	gcloud run jobs execute scrape-netkeiba-job-race-dates \
		--args="--year,$(YEAR),--month,$(MONTH)" \
		--region=$(REGION) \
		$(GLOUD_WIDE_FLAG)

# Required:
#  - TRAIN_FIRST_DATE: e.g. 20240101
#  - TRAIN_LAST_DATE: e.g. 20241031
#  - VALID_LAST_DATE: e.g. 20250301
#  - DATA_VERSION: e.g. 20250315_100000
.PHONY: execute_job_train_lgb
execute_job_train_lgb:
	gcloud run jobs execute train-lgb-job \
		--args="--train-first-date,$(TRAIN_FIRST_DATE),--train-last-date,$(TRAIN_LAST_DATE),--valid-last-date,$(VALID_LAST_DATE),--data-version,$(DATA_VERSION)" \
		--region=$(REGION) \
		$(GLOUD_WIDE_FLAG)

# Required:
#  - TRAIN_FIRST_DATE: e.g. 20240101
#  - TRAIN_LAST_DATE: e.g. 20241031
#  - VALID_LAST_DATE: e.g. 20250301
#  - DATA_VERSION: e.g. 20250315_100000
.PHONY: execute_job_train_xgb
execute_job_train_xgb:
	gcloud run jobs execute train-xgb-job \
		--args="--train-first-date,$(TRAIN_FIRST_DATE),--train-last-date,$(TRAIN_LAST_DATE),--valid-last-date,$(VALID_LAST_DATE),--data-version,$(DATA_VERSION)" \
		--region=$(REGION) \
		$(GLOUD_WIDE_FLAG)

.PHONY: nb
nb:
	uv run jupyter-lab
