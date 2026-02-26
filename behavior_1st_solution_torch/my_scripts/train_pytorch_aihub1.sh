export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# 使用你刚才的端口和 IP
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29501 \
    --rdzv_id=test_dist \
    check_dist.py
