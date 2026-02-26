uv run --no-sync scripts/train.py pi_behavior_b1k_fast_debug \
	  --batch_size=16 \
	  --num_train_steps=200000 \
	  --fsdp_devices=8 \
	  --save_interval=250 \
		--keep_period=4000 \
		--log_interval=25 \
		--overwrite
