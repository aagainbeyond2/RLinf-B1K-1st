import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import interp1d
import numpy as np

def read_tensorboard_data(log_dir, tag_name):
    """
    读取 TensorBoard 日志文件中的标量数据
    """
    # 找到目录下的 tfevents 文件
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"Warning: No tfevents file found in {log_dir}")
        return pd.DataFrame({"step": [], "value": []})

    # 默认读取最新的一个文件
    event_file = max(event_files, key=os.path.getctime)
    print(f"Loading log: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag_name not in ea.Tags()['scalars']:
        available = ea.Tags()['scalars']
        # 尝试模糊匹配，因为有时候 tag 会带有前缀
        candidates = [t for t in available if tag_name in t]
        if candidates:
            print(f"Tag '{tag_name}' not found exactly, using '{candidates[0]}' instead.")
            tag_name = candidates[0]
        else:
            raise ValueError(f"Tag '{tag_name}' not found. Available tags: {available}")

    events = ea.Scalars(tag_name)

    steps = [x.step for x in events]
    values = [x.value for x in events]

    return pd.DataFrame({"step": steps, "value": values})

def plot_comparison(
    jax_log_dir,
    pt_log_dir,
    jax_tag="train_loss/action_loss",
    pt_tag="train_loss/action_loss"
):
    # ================= 配置区 =================
    # PyTorch 配置
    PT_BATCH_SIZE = 64
    PT_ACCUM_STEPS = 32

    # JAX 配置
    JAX_BATCH_SIZE = 1024
    JAX_ACCUM_STEPS = 2

    # Y轴对齐策略: 以 PyTorch (真实 Mean) 为基准
    # JAX Log 被人为乘了 JAX_ACCUM_STEPS (2)，所以我们需要除以 2 来还原
    JAX_LOSS_SCALE = 1.0 / JAX_ACCUM_STEPS  # 0.5
    PT_LOSS_SCALE = 1.0                     # 保持不变
    # =========================================

    # 1. 读取数据
    print(f"--- Reading JAX logs ({jax_tag}) ---")
    df_jax = read_tensorboard_data(jax_log_dir, jax_tag)

    print(f"--- Reading PyTorch logs ({pt_tag}) ---")
    df_pt = read_tensorboard_data(pt_log_dir, pt_tag)

    if df_jax.empty or df_pt.empty:
        print("Error: One of the dataframes is empty. Check your paths and tags.")
        return

    # 2. X 轴对齐：转换为 "Total Samples Seen"
    # PyTorch: step * 64
    df_pt["total_samples"] = df_pt["step"] * PT_BATCH_SIZE

    # JAX: step * 1024
    df_jax["total_samples"] = df_jax["step"] * JAX_BATCH_SIZE
    import pdb;pdb.set_trace()
    max_pt_samples = df_pt["total_samples"].max()
    print(f"\nTruncating JAX data to match PyTorch length: {max_pt_samples} samples")

    # 保留那些小于等于 PyTorch 最大样本数的 JAX 数据点
    df_jax = df_jax[df_jax["total_samples"] <= max_pt_samples]

    # 3. Y 轴对齐 (JAX 除以 2)
    df_pt["value_scaled"] = df_pt["value"] * PT_LOSS_SCALE
    df_jax["value_scaled"] = df_jax["value"] * JAX_LOSS_SCALE

    # 4. 绘图
    plt.figure(figsize=(12, 7))

    # 绘制 JAX 曲线
    plt.plot(
        df_jax["total_samples"],
        df_jax["value_scaled"],
        label=f"JAX (Scaled x{JAX_LOSS_SCALE:.2f} -> Real Mean)",
        alpha=0.7,
        linewidth=1.5,
        color='tab:orange'
    )

    # 绘制 PyTorch 曲线
    plt.plot(
        df_pt["total_samples"],
        df_pt["value_scaled"],
        label=f"PyTorch (Original Real Mean)",
        alpha=0.8,
        linewidth=1.5,
        linestyle='--',
        color='tab:blue'
    )

    plt.title(f"Loss Comparison: JAX vs PyTorch (Aligned to Real Mean)\nTag: {jax_tag}", fontsize=14)
    plt.xlabel("Total Samples Seen (Global Batch Size = 2048)", fontsize=12)
    plt.ylabel("Loss (Real Mean Value)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    # 添加辅助 X 轴：显示 Optimizer Updates
    def samples_to_updates(x):
        return x / 2048 # 2048 is Global Batch Size

    def updates_to_samples(x):
        return x * 2048

    secax = plt.gca().secondary_xaxis('top', functions=(samples_to_updates, updates_to_samples))
    secax.set_xlabel('Optimizer Update Steps (Global BS=2048)', fontsize=10)

    # 5. 计算并打印对齐后的统计信息 (可选)
    # 取后半段数据计算平均值，看收敛位置是否接近
    start_sample = min(df_pt["total_samples"].max(), df_jax["total_samples"].max()) * 0.5

    jax_final_mean = df_jax[df_jax["total_samples"] > start_sample]["value_scaled"].mean()
    pt_final_mean = df_pt[df_pt["total_samples"] > start_sample]["value_scaled"].mean()

    print("\n--- Statistics (Last 50% of training) ---")
    print(f"JAX Mean Loss: {jax_final_mean:.5f}")
    print(f"PT  Mean Loss: {pt_final_mean:.5f}")
    print(f"Difference:    {abs(jax_final_mean - pt_final_mean):.5f}")

    plt.tight_layout()
    output_file = "loss_comparison_real_mean.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")
    plt.show()

# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    # 请修改这里的路径为你实际的文件夹路径
    # 文件夹内应包含 events.out.tfevents.xxxx 文件
    JAX_DIR = "/ms/PA/yejiaquan/b1k_models/checkpoints/pi_behavior_b1k_fast_stage2/openpi/tensorboard"
    PT_DIR = "/ms/PA/yejiaquan/b1k_models/checkpoints/pytorch_pi_behavior_b1k_fast_stage2_gradAccum32_V2/openpi/tensorboard"

    # 确保 Tag 名字正确，根据你的截图，应该是 "train_loss/action_loss"
    # 或者是 "train/action_loss" (取决于你具体的代码版本)
    JAX_TAG = "train/action_loss"
    PT_TAG = "train_loss/action_loss"

    if os.path.exists(JAX_DIR) and os.path.exists(PT_DIR):
        plot_comparison(JAX_DIR, PT_DIR, jax_tag=JAX_TAG, pt_tag=PT_TAG)
    else:
        print("Please set JAX_DIR and PT_DIR to valid paths in the script.")