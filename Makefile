.PHONY: execute_job_scrape_netkeiba
execute_job_scrape_netkeiba:
	gcloud run jobs execute scrape-netkeiba-job \
		--args="--race-date,$(RACE_DATE)" \
		--region=asia-northeast1
