## Run B1K 1st Rollout
```bash
export ISAAC_PATH=/data1/heyelin/rlinf/isaac-sim
export OMNIGIBSON_DATA_PATH=/data1/heyelin/BEHAVIOR-1K-main/datasets
export RLINF_DEBUG_SHAPES=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export RLINF_DEBUG_OPENPI=1
bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi_pi05_debug_rollout_only_3x3_b1k_tricks rollout.save_rollout.dir=./offline_rollouts env.train.video_cfg.save_video=true env.train.video_cfg.video_base_dir=./videos

