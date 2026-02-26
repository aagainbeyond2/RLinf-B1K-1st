"""
PyTorch Training script for BEHAVIOR-1K solution.
Supports Multi-Node HSDP (Hybrid Sharded Data Parallel):
- Intra-node: FSDP (Sharding)
- Inter-node: DDP (Replication)
"""

import dataclasses
import logging
import os
import platform
import time
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh

import wandb
import tqdm
import dill

from b1k.models.pi_behavior_config import PiBehaviorConfig
from openpi.models_pytorch.pi_behavior_pytorch import PiBehaviorPytorch
#from b1k.training import config as _config
from b1k.training import pytorch_config as _config
from b1k.training import data_loader as _data_loader

from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma import modeling_gemma

import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from b1k.training import data_loader as _b1k_loader
from b1k.models.observation import Observation

import contextlib
import warnings
warnings.filterwarnings("ignore", message="The video decoding and encoding capabilities of torchvision are deprecated")
from torch.utils.tensorboard import SummaryWriter
import math
import psutil
import gc
import multiprocessing



# ==============================================================================
# Utils
# ==============================================================================

_proc = psutil.Process(os.getpid())

def log_mem(prefix, pbar):
    vm = psutil.virtual_memory()
    children = _proc.children(recursive=True)
    rss_self = _proc.memory_info().rss / (1024**3)
    rss_child = sum((c.memory_info().rss for c in children), 0) / (1024**3)
    pbar.write(
        f"{prefix} | SYS used={vm.used/1024**3:.1f}G avail={vm.available/1024**3:.1f}G cached={getattr(vm,'cached',0)/1024**3:.1f}G "
        f"| RSS self={rss_self:.1f}G workers={rss_child:.1f}G nchild={len(children)}"
    )


class PiBehaviorTrainingWrapper(nn.Module):
    """
    make FSDP happy
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observation, actions, train=True):
        return self.model.forward_detailed(observation, actions, train=train)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

def setup_distributed():
    """Initialize distributed training environment."""
    # torchrun 会自动设置这些环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank

def init_logging(rank):
    """Setup logging only on rank 0."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        datefmt="%H:%M:%S"
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def custom_collate_fn(batch_list):
    def stack_and_convert(key_values):
        if isinstance(key_values[0], (np.ndarray, float, int, np.number, np.bool_, bool)):
            batched = np.stack(key_values)
            t = torch.from_numpy(batched)
            if t.dtype == torch.float64:
                t = t.float()
            return t
            
        elif isinstance(key_values[0], dict):
            return {
                k: stack_and_convert([x[k] for x in key_values])
                for k in key_values[0].keys()
            }
            
        elif isinstance(key_values[0], (list, tuple)):
            transposed = zip(*key_values)
            return type(key_values[0])([stack_and_convert(list(col)) for col in transposed])

        return key_values

    return stack_and_convert(batch_list)

def build_b1k_dataloader(config, rank, world_size):
    """
    PyTorch Distributed DataLoader
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = _b1k_loader.create_behavior_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        seed=config.seed
    )

    dataset = _b1k_loader.transform_dataset(dataset, data_config)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True, # 丢弃最后不足一个 batch 的数据，保证 shape 固定
        seed=config.seed if config.seed is not None else 0
    )

    local_batch_size = config.batch_size // world_size

    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        multiprocessing_context=multiprocessing.get_context("spawn"),
    )

    return loader, data_config.norm_stats


# ==============================================================================
# Knowledge Insulation Monitoring (Gradient Norms)
# ==============================================================================

def compute_group_grad_norm(model, predicate):
    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and predicate(name):
            grads.append(param.grad.detach().flatten())

    if grads:
        full_grad = torch.cat(grads)
        return torch.norm(full_grad, p=2).item()
    return 0.0

def is_action_expert_param(name):
    keywords = [
        "gemma_expert", # 对应 JAX 的 _1 expert
        "action_in_proj",
        "action_out_proj",
        "time_mlp",
        "kv_transform"
    ]
    return any(k in name for k in keywords)

def is_vlm_param(name):
    return not is_action_expert_param(name)

# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    # 1. Setup Distributed
    rank, world_size, local_rank = setup_distributed()
    init_logging(rank)

    config = _config.cli()

    device = torch.device(f"cuda:{local_rank}")

    seed = 1234
    set_seed(seed + rank) # 不同 rank 不同 seed 用于数据 shuffle

    if rank == 0:
        logging.info(f"Running on: {platform.node()}, World Size: {world_size}")

    # 2. Setup Device Mesh for HSDP (Hybrid Sharding)
    num_gpus_per_node = torch.cuda.device_count()
    nnodes = world_size // num_gpus_per_node

    if nnodes > 1:
        # HSDP 
        device_mesh = init_device_mesh("cuda", (nnodes, num_gpus_per_node), mesh_dim_names=("dp", "fsdp"))
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        if rank == 0:
            logging.info(f"Using HSDP (Hybrid Sharded Data Parallel). Nodes: {nnodes}, GPUs/Node: {num_gpus_per_node}")
    else:
        # FSDP
        device_mesh = None
        sharding_strategy = ShardingStrategy.FULL_SHARD
        if rank == 0:
            logging.info("Using Standard FSDP (Single Node).")

    train_loader, norm_stats = build_b1k_dataloader(config, rank, world_size)
    data_iter = iter(train_loader)

    # tensorboard
    if rank == 0:
        tb_dir = os.path.join(config.checkpoint_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        # init TensorBoard Writer
        writer = SummaryWriter(log_dir=tb_dir)
        logging.info(f"TensorBoard logging to: {tb_dir}")

    # 3. Model Init
    model = PiBehaviorPytorch(config.model)
    model.to(torch.bfloat16)

    # Load Norm Stats (Correlation Matrix)
    if hasattr(model, 'load_correlation_matrix') and norm_stats is not None:
         model.load_correlation_matrix(norm_stats)

    model.to(device)


    # Load Pretrained Checkpoint (Full/Non-Sharded)
    if hasattr(config, 'pytorch_weight_path') and config.pytorch_weight_path:

        if rank == 0:
            logging.info(f"Loading pretrained full checkpoint from: {config.pytorch_weight_path}")

        if os.path.exists(config.pytorch_weight_path):
            checkpoint_data = torch.load(config.pytorch_weight_path, map_location='cpu')

            if 'model' in checkpoint_data:
                state_dict = checkpoint_data['model']
            elif 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
            elif 'params' in checkpoint_data:
                state_dict = checkpoint_data['params']
            else:
                state_dict = checkpoint_data  # 假设整个文件就是 state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

            if rank == 0:
                logging.info("Pretrained weights loaded successfully.")
                if missing_keys:
                    logging.warning(
                        f"Missing keys (initialized randomly): {missing_keys[:5]} ... ({len(missing_keys)} total)")
                if unexpected_keys:
                    logging.warning(
                        f"Unexpected keys (ignored): {unexpected_keys[:5]} ... ({len(unexpected_keys)} total)")

                del state_dict
                del checkpoint_data
        else:
            if rank == 0:
                raise FileNotFoundError(f"Checkpoint not found: {config.pytorch_weight_path}")

    if rank == 0:
        logging.info("Wrapping model with PiBehaviorTrainingWrapper for FSDP compatibility...")
    model = PiBehaviorTrainingWrapper(model)

    # 4. FSDP Wrapping
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32, # Reduce 用 FP32 保持精度
        buffer_dtype=torch.bfloat16,
    )

    # Wrap Policy: 告诉 FSDP 哪些层需要单独 wrap (通常是 Transformer Block)
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            SiglipEncoderLayer,  
            GemmaDecoderLayer, 
        },
    )

    model = FSDP(
        model,
        device_mesh=device_mesh, 
        sharding_strategy=sharding_strategy,
        mixed_precision=bf16_policy,
        auto_wrap_policy=my_auto_wrap_policy,
        device_id=device,
        use_orig_params=True, 
        sync_module_states=True, 
    )

    # ==============================================================================
    # EMA Setup
    # ==============================================================================
    ema_model = None
    if config.ema_decay is not None and config.ema_decay < 1.0:
        if rank == 0:
            logging.info(f"⚡️ Initializing EMA model with decay: {config.ema_decay}")

        ema_inner = PiBehaviorPytorch(config.model)
        ema_inner.to(torch.bfloat16)

        if hasattr(ema_inner, 'load_correlation_matrix') and norm_stats is not None:
             ema_inner.load_correlation_matrix(norm_stats)
        ema_inner.to(device)

        ema_model = PiBehaviorTrainingWrapper(ema_inner)

        ema_model = FSDP(
            ema_model,
            device_mesh=device_mesh,
            sharding_strategy=sharding_strategy,
            mixed_precision=bf16_policy,
            auto_wrap_policy=my_auto_wrap_policy, # 复用同一个 wrap policy
            device_id=device,
            use_orig_params=True,
            sync_module_states=True,
        )

        with torch.no_grad():
            for p_train, p_ema in zip(model.parameters(), ema_model.parameters()):
                p_ema.data.copy_(p_train.data)
                p_ema.requires_grad_(False) # EMA 永远不需要梯度

        if rank == 0:
            logging.info("✅ EMA model initialized and synchronized from Main Model.")



    # 5. Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr, # 适配 config 结构
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    def lr_lambda(step):
        # 1. Warmup Phase
        if step < config.lr_schedule.warmup_steps:
            # 线性增长: step / warmup_steps
            return float(step) / float(max(1, config.lr_schedule.warmup_steps))

        # 2. Cosine Decay Phase
        # 进度: 0.0 (刚结束 warmup) -> 1.0 (完成 decay)
        progress = float(step - config.lr_schedule.warmup_steps) / float(max(1, config.lr_schedule.decay_steps))

        # 如果超过了 decay_steps，保持在最低 LR (或者 0，视具体需求，这里保持 decay_lr)
        if progress >= 1.0:
            return config.lr_schedule.decay_lr / config.lr_schedule.peak_lr

        # Cosine 计算: 0.5 * (1 + cos(pi * progress))
        # 范围从 1.0 降到 0.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # 缩放范围: [decay_lr/peak_lr, 1.0]
        # 公式: min + (max - min) * decay_factor
        min_ratio = config.lr_schedule.decay_lr / config.lr_schedule.peak_lr
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # 7. Checkpoint Loading (Resume)
    start_step = 0
    if config.resume:
        pass

    # 8. Training Loop
    if rank == 0:
        pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step, total=config.num_train_steps)

    model.train()

    # get grad_accum_steps
    grad_accum_steps = getattr(config, 'grad_accum_steps', 1)

    if rank == 0:
        logging.info(f"Training with Gradient Accumulation Steps: {grad_accum_steps}")
        logging.info(f"Total Micro Steps (Batches): {config.num_train_steps}")
        logging.info(f"Total Optimizer Updates: {config.num_train_steps // grad_accum_steps}")


    accum_loss_tracker = 0.0
    detailed_loss_tracker = {}
    grad_norm = torch.tensor(0.0, device=device)

    for step in range(start_step, config.num_train_steps):
        t0 = time.time()

        # A. Get Batch & Move to Device
        try:
            batch = next(data_iter)
        except StopIteration:
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(step)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        is_update_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == config.num_train_steps)

        if step == start_step and rank == 0:
            actions_data = batch["actions"]
            logging.info(f"    >>> Action shape: {actions_data.shape}")
            logging.info(f"    >>> Local batch size: {actions_data.shape[0]}")

        def to_device(item):
            if isinstance(item, torch.Tensor):
                return item.to(device, non_blocking=True)
            elif isinstance(item, dict):
                return {k: to_device(v) for k, v in item.items()}
            return item

        batch = to_device(batch)
        actions = batch.pop("actions")  # 提取 Action GT
        observation = Observation.from_dict(batch)

        #with sync_context:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            #losses_dict = model.forward_detailed(observation, actions, train=True)
            losses_dict = model(observation, actions, train=True)
            total_loss = losses_dict["total_loss"]

            # for smooth loss log
            accum_loss_tracker += total_loss.item()
            for k, v in losses_dict.items():
                if k == "total_loss": continue
                val = v.item() if isinstance(v, torch.Tensor) else v
                if k not in detailed_loss_tracker:
                    detailed_loss_tracker[k] = 0.0
                detailed_loss_tracker[k] += val

            scaled_loss = total_loss / grad_accum_steps
            scaled_loss.backward()

        if is_update_step:
            #　gradient clip
            if config.max_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            else:
                grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in model.parameters() if p.grad is not None]),
                    2.0
                )

            # D. Optimizer Step
            optimizer.step()
            scheduler.step()

            # EMA update
            if ema_model is not None:
                decay = config.ema_decay
                with torch.no_grad():
                    # new_avg = decay * old_avg + (1 - decay) * new_val
                    for p_train, p_ema in zip(model.parameters(), ema_model.parameters()):
                        p_ema.data.mul_(decay).add_(p_train.data, alpha=(1.0 - decay))

            optimizer.zero_grad()

            # 5. Logging 
            if rank == 0:
                avg_total_loss = accum_loss_tracker / grad_accum_steps

                avg_detailed_losses = {
                    k: v / grad_accum_steps
                    for k, v in detailed_loss_tracker.items()
                }

                lr_value = optimizer.param_groups[0]['lr']
                grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                # --- TensorBoard ---
                writer.add_scalar("train/total_loss", avg_total_loss, step)
                writer.add_scalar("train/grad_norm", grad_norm_value, step)
                writer.add_scalar("train/lr", lr_value, step)

                for k, v in avg_detailed_losses.items():
                    if "accuracy" in k:
                        writer.add_scalar(f"train_metrics/{k}", v, step)
                    else:
                        writer.add_scalar(f"train_loss/{k}", v, step)

                # logging memory
                mem_info = psutil.virtual_memory()
                used_gb = mem_info.used / (1024 ** 3)
                total_gb = mem_info.total / (1024 ** 3)
                # TensorBoard
                writer.add_scalar("sys/ram_used_gb", used_gb, step)

                # --- 控制台打印 ---
                action_keys = []
                special_keys = []

                for k, v in avg_detailed_losses.items():
                    if k.startswith("action_loss_"):
                        short_name = k.replace("action_loss_", "")
                        action_keys.append(f"{short_name}={v:.4f}")
                    else:
                        special_keys.append(f"{k}={v:.4f}")

                header_msg = f"[Step {step}] Avg_Loss={avg_total_loss:.4f} | G={grad_norm_value:.2f} | LR={lr_value:.2e}"
                if special_keys:
                    header_msg += " | " + " | ".join(special_keys)

                pbar.write(header_msg)

                if action_keys:
                    chunk_size = 5
                    for i in range(0, len(action_keys), chunk_size):
                        pbar.write("    Actions: " + "  ".join(action_keys[i:i+chunk_size]))

                pbar.set_description(f"Loss: {avg_total_loss:.4f} | RAM: {used_gb:.1f}/{total_gb:.1f}GB")

            accum_loss_tracker = 0.0
            detailed_loss_tracker = {}

            if rank == 0:
                mem_before = psutil.virtual_memory().used / (1024 ** 3)
            # gc.collect()  # 所有 Rank 都要执行
            if rank == 0:
                mem_after = psutil.virtual_memory().used / (1024 ** 3)
                diff = mem_before - mem_after
                # 使用 pbar.write 防止被进度条覆盖
                pbar.write(
                    f"[System] GC triggered at step {step}. "
                    f"RAM: {mem_before:.2f}GB -> {mem_after:.2f}GB (Freed: {diff:.2f}GB)"
                )

            if rank == 0:
                log_mem("[LOG MEM]", pbar)

        if rank == 0:
            pbar.update(1)



        # F. Save Checkpoint
        if step > 0 and step % config.save_interval == 0:

            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                model_state = model.state_dict()
                optim_state = FSDP.optim_state_dict(model, optimizer)

            ema_state = None
            if ema_model is not None:
                with FSDP.state_dict_type(ema_model, StateDictType.SHARDED_STATE_DICT):
                    ema_state = ema_model.state_dict()
                    
            save_path = os.path.join(config.checkpoint_dir, f"ckpt_{step}_rank{rank}.pt")

            save_dict = {
                'step': step,
                'model_shard': model_state,
                'optimizer_shard': optim_state, # 包含了 momentum 等状态的分片
                'config': dataclasses.asdict(config)
            }

            if ema_state is not None:
                save_dict["ema_model_shard"] = ema_state
                
            torch.save(save_dict, save_path)
            
            if rank == 0:
                logging.info(f"Saved SHARDED checkpoint to {config.checkpoint_dir}/ckpt_{step}_rank*.pt")

            dist.barrier()


    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()