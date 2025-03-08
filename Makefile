.PHONY: train

# Required
#  - DT
#  - RACE_ID
train:
	uv run python src/horse_racing/app/train.py \
		--dt $(DT) \
		--race_id $(RACE_ID)
