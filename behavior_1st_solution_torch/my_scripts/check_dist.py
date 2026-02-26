import os
import torch
import torch.distributed as dist

def main():
    print(f"[Start] Rank {os.environ.get('RANK')} is starting...", flush=True)

    # 1. 初始化
    try:
        dist.init_process_group(backend="nccl")
        print(f"[Init] Rank {dist.get_rank()} process group initialized.", flush=True)
    except Exception as e:
        print(f"[Error] Init failed: {e}", flush=True)
        return

    # 2. 获取设备
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Cuda] Rank {dist.get_rank()} using device {device}", flush=True)

    # 3. 简单的 Tensor 计算
    tensor = torch.ones(1).to(device)
    print(f"[Tensor] Rank {dist.get_rank()} created tensor.", flush=True)

    # 4. 集合通信 (AllReduce)
    try:
        dist.all_reduce(tensor)
        print(f"[NCCL] Rank {dist.get_rank()} AllReduce successful! Value: {tensor.item()}", flush=True)
    except Exception as e:
        print(f"[Error] NCCL failed: {e}", flush=True)

    # 5. 结束
    dist.destroy_process_group()
    print(f"[End] Rank {dist.get_rank()} finished.", flush=True)

if __name__ == "__main__":
    main()
