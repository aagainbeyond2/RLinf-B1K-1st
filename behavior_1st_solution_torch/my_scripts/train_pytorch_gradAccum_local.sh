export NCCL_SOCKET_IFNAME=front1
export NCCL_IB_DISABLE=1

torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch_gradAccum.py pytorch_pi_behavior_b1k_fast_stage2_gradAccum32 --batch_size 64 --grad_accum_steps 32
# debug fast load
#torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch_gradAccum.py pytorch_pi_behavior_b1k_fast_stage2 --batch_size 4
