export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

#torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch.py pytorch_pi_behavior_b1k_fast_stage2 --batch_size 4
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pytorch_pi_behavior_b1k_fast_stage2 --batch_size 64
