export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET


torchrun --nnodes=1 --nproc_per_node=8 scripts/train_pytorch_ac.py pytorch_pi_behavior_b1k_fast_stage2_activation_checkpoint --batch_size 64 --grad_accum_steps 32
