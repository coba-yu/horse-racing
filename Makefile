REGION=asia-northeast1

# Required:
#  - RACE_DATE: e.g. 20210101
.PHONY: execute_job_scrape_netkeiba
execute_job_scrape_netkeiba:
	gcloud run jobs execute scrape-netkeiba-job \
		--args="--race-date,$(RACE_DATE)" \
		--region=$(REGION)

# Required:
#  - RACE_DATES: e.g. 20210101,20210102
.PHONY: execute_workflow_scrape_netkeiba
execute_workflow_scrape_netkeiba:
	gcloud workflows execute scrape-netkeiba \
		--location=$(REGION) \
		--data='{"race_dates": [$(RACE_DATES)]}'

# Required:
#  - YEAR: e.g. 2021
#  - MONTH: e.g. 1
.PHONY: execute_job_scrape_netkeiba_race_dates
execute_job_scrape_netkeiba_race_dates:
	gcloud run jobs execute scrape-netkeiba-job-race-dates \
		--args="--year,$(YEAR),--month,$(MONTH)" \
		--region=$(REGION)
