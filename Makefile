.PHONY: check data smoke smoke-train

check:
	python scripts/check_release.py

data:
	bash scripts/prepare_data.sh

smoke:
	bash scripts/smoke_train_and_attack.sh

smoke-train:
	SMOKE_RUN_HERETIC=false bash scripts/smoke_train_and_attack.sh
