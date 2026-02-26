export NCCL_SOCKET_IFNAME=front1
export NCCL_IB_DISABLE=1

#torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch.py pytorch_pi_behavior_b1k_fast_stage2
torchrun --standalone --nnodes=1 --nproc_per_node=8 --node_rank=$RANK --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} scripts/train_pytorch.py pytorch_pi_behavior_b1k_fast_stage2 --batch_size 64
