# 强杀所有 python 进程，确保环境干净
pkill -9 python
pkill -9 python3

export NCCL_DEBUG=INFO

# 1. 环境变量 (保持之前的网络设置)
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# [新增] 强制 Python 实时输出日志，不再缓冲
export PYTHONUNBUFFERED=1
# [新增] 打印分布式调试信息
export TORCH_DISTRIBUTED_DEBUG=INFO

# 2. 启动命令
# 注意：端口改成了 29501，id 改成了 job_new
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29501 \
    --rdzv_id=job_new_port \
    scripts/train_pytorch.py pytorch_pi_behavior_b1k_fast_stage2
