.PHONY: train
train:
	uv run python src/horse_racing/app/train.py

# Required
#  - DT
#  - RACE_ID
#  - VERSION
#  - LGB_OBJECTIVE
.PHONY: predict
predict:
	uv run python src/horse_racing/app/predict.py \
		--dt $(DT) \
		--race_id $(RACE_ID) \
		--version $(VERSION) \
		--lgb-objective $(LGB_OBJECTIVE)
