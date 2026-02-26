export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
#export NCCL_SOCKET_IFNAME=front1
#export NCCL_IB_DISABLE=1


#torchrun --standalone --nnodes=1 --nproc_per_node=8 --node_rank=$RANK --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} scripts/train_pytorch_gradAccum.py pytorch_pi_behavior_b1k_fast_stage2_gradAccum32_debug --batch_size 64 --grad_accum_steps 32
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_pytorch_gradAccum.py pytorch_pi_behavior_b1k_fast_stage2_gradAccum32_debug --batch_size 64 --grad_accum_steps 32
