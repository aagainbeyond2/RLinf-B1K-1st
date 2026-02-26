import torch
import numpy as np
import dill
import os
import jax.numpy as jnp
from types import SimpleNamespace
from flax.traverse_util import flatten_dict, unflatten_dict

# 导入你的 PyTorch 模型定义
import sys
sys.path.append("/data1/heyelin/rlinf/RLinf-main/behavior_1st_solution_torch/openpi/src/openpi/models_pytorch/")
from pi_behavior_pytorch import PiBehaviorPytorch

# 路径配置
REPRODUCE_DIR = "/data1/heyelin/models/behavior_1st_torch/checkpoint_2/"

def jax_params_to_pytorch_state_dict(jax_params, pt_model):
    """
    核心转换函数：将 JAX 参数字典转换为 PyTorch state_dict
    1. 扁平化 JAX 参数树
    2. 转换名称 (Regex替换)
    3. 转换张量 (转置 Linear 权重, 转换 LayerNorm scale)
    """
    print("Converting JAX params to PyTorch format...")
    pt_state_dict = pt_model.state_dict()
    new_state_dict = {}

    # 扁平化 JAX 参数，key 变成 tuple ('PaliGemma', 'llm', ...)
    flat_jax = flatten_dict(jax_params, sep='.')

    # 打印一些 Keys 用于调试映射规则
    # print("Sample JAX Keys:", list(flat_jax.keys())[:5])

    for jax_key, jax_val in flat_jax.items():
        # 转为 numpy
        val = np.array(jax_val)

        # --- 1. 名称映射规则 (需要根据你的 pytorch 模型命名调整) ---
        pt_key = jax_key

        # 基础替换
        pt_key = pt_key.replace("kernel", "weight")
        pt_key = pt_key.replace("scale", "weight") # LayerNorm
        pt_key = pt_key.replace("embedding", "weight")

        # 模块路径映射 (JAX -> PyTorch)
        # 注意：这里的映射必须严格对应 PiBehaviorPytorch 的层级结构
        pt_key = pt_key.replace("PaliGemma.llm", "paligemma_with_expert.paligemma.language_model.model")
        pt_key = pt_key.replace("PaliGemma.img", "paligemma_with_expert.paligemma.vision_tower")

        # Action Expert 映射
        pt_key = pt_key.replace("action_in_proj", "action_in_proj")
        pt_key = pt_key.replace("action_out_proj", "action_out_proj")
        pt_key = pt_key.replace("time_mlp_in", "time_mlp_in")
        pt_key = pt_key.replace("time_mlp_out", "time_mlp_out")

        # Transformer Layer 映射 (Flax 通常是 layers_0, PyTorch 是 layers.0)
        # 使用简单的字符串替换可能不够，建议用正则，这里简化处理
        pt_key = pt_key.replace("layers_", "layers.")

        # Attention 映射 (Flax: key/kernel -> PyTorch: k_proj.weight)
        # 注意：Gemma PyTorch 实现通常是 q_proj, k_proj, v_proj, o_proj
        # 但 Flax 实现可能是 combined qkv 或者分开的
        # 假设这里也是分开的，如果名字不匹配需要手动修

        # --- 2. 张量形状处理 ---
        # 关键：JAX Linear 是 (In, Out)，PyTorch 是 (Out, In) -> 需要转置
        # 识别 Linear 层权重的特征：通常以 .weight 结尾且维度为 2
        if pt_key.endswith(".weight") and val.ndim == 2:
            # 只有 Linear 层需要转置，Embedding 层不需要 (虽然也是2维)
            # Embedding 通常在 PyTorch 中是 (Num, Dim)，JAX 也是 (Num, Dim)，无需转置
            # 区分方法：看名字是否包含 embedding，或者检查 PyTorch 对应层的类型

            # 为了安全，我们检查 PyTorch 模型中对应 key 的形状
            if pt_key in pt_state_dict:
                target_shape = pt_state_dict[pt_key].shape
                if tuple(val.T.shape) == target_shape:
                    val = val.T
                elif tuple(val.shape) == target_shape:
                    pass # 形状已匹配 (如 Embedding)
                else:
                    print(f"Warning: Shape mismatch for {pt_key}. JAX: {val.shape}, PT Target: {target_shape}")
            else:
                # 尝试模糊匹配或忽略
                # print(f"Skipping {pt_key} (not found in PT model)")
                continue

        # --- 3. 赋值 ---
        if pt_key in pt_state_dict:
            new_state_dict[pt_key] = torch.from_numpy(val)
        else:
            # 调试用：打印未匹配的键
            # print(f"Unmatched JAX key: {jax_key} -> {pt_key}")
            pass

    return new_state_dict

class ObsAdapter:
    """
    将 JAX 的 Observation 对象转换为 PyTorch 模型可接受的对象格式。
    PyTorch 模型通过访问该对象的属性 (如 .images, .state) 来获取数据。
    """
    def __init__(self, jax_obs, device):
        self.images = {}
        self.image_masks = {}

        # --- 1. 处理图片 (含 5D 修复) ---
        for k, v in jax_obs.images.items():
            img_np = np.array(v)
            if img_np.ndim == 4:
                img_np = np.expand_dims(img_np, axis=1)
            img_pt = torch.from_numpy(img_np).permute(0, 1, 4, 2, 3).to(device)
            self.images[k] = img_pt.float()

        # --- 2. 处理图片 Mask (含 5D 修复) ---
        for k, v in jax_obs.image_masks.items():
            mask_np = np.array(v)
            if mask_np.ndim == 1:
                mask_np = np.expand_dims(mask_np, axis=1)
            self.image_masks[k] = torch.from_numpy(mask_np).to(device)

        # --- 3. 处理状态和 Prompt ---
        self.state = torch.from_numpy(np.array(jax_obs.state)).to(device).float()
        self.tokenized_prompt = torch.from_numpy(np.array(jax_obs.tokenized_prompt)).to(device)

        # 处理 tokenized_prompt_mask
        if hasattr(jax_obs, 'tokenized_prompt_mask') and jax_obs.tokenized_prompt_mask is not None:
            self.tokenized_prompt_mask = torch.from_numpy(np.array(jax_obs.tokenized_prompt_mask)).to(device)
        else:
            self.tokenized_prompt_mask = torch.ones_like(self.tokenized_prompt, dtype=torch.bool, device=device)

        # [修复]: 补充 token_ar_mask
        if hasattr(jax_obs, 'token_ar_mask') and jax_obs.token_ar_mask is not None:
            self.token_ar_mask = torch.from_numpy(np.array(jax_obs.token_ar_mask)).to(device)
        else:
            self.token_ar_mask = None # 如果源数据没有，设为None

        # [修复]: 预防性补充 token_loss_mask (pdb中显示也有这个字段)
        if hasattr(jax_obs, 'token_loss_mask') and jax_obs.token_loss_mask is not None:
            self.token_loss_mask = torch.from_numpy(np.array(jax_obs.token_loss_mask)).to(device)
        else:
            self.token_loss_mask = None

        # --- 4. 处理 FAST Tokens (可选) ---
        if hasattr(jax_obs, 'fast_tokens') and jax_obs.fast_tokens is not None:
            self.fast_tokens = torch.from_numpy(np.array(jax_obs.fast_tokens)).to(device)
            self.fast_token_mask = torch.from_numpy(np.array(jax_obs.fast_token_mask)).to(device)
        else:
            self.fast_tokens = None
            self.fast_token_mask = None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载 Config 和 Data (从之前 JAX 运行保存的文件)
    print("Loading JAX artifacts...")
    with open(os.path.join(REPRODUCE_DIR, "config_from_jax.pkl"), "rb") as f:
        jax_config = dill.load(f)

    with open(os.path.join(REPRODUCE_DIR, "batch_from_jax.pkl"), "rb") as f:
        jax_batch = dill.load(f)
        jax_obs, jax_actions = jax_batch

    with open(os.path.join(REPRODUCE_DIR, "data_config_from_jax.pkl"), "rb") as f:
        data_config = dill.load(f)

    # 2. 初始化 PyTorch 模型
    print("Initializing PyTorch model...")
    if hasattr(jax_config, 'model'):
        model_config = jax_config.model
    else:
        model_config = jax_config

    model = PiBehaviorPytorch(model_config)
    model.to(device)
    model.eval()

    # 3. 准备 PyTorch 输入
    print("Adapting inputs...")
    pt_obs = ObsAdapter(jax_obs, device) # 确保 ObsAdapter 已经改成了 bf16

    # [Debug] 检查第一张图片的数值范围
    first_img_key = list(pt_obs.images.keys())[0]
    img_tensor = pt_obs.images[first_img_key]
    print(f"[DEBUG] Image Range: Min={img_tensor.min().item()}, Max={img_tensor.max().item()}")

    pt_actions = torch.from_numpy(np.array(jax_actions)).to(device).float()
    B, H, D = pt_actions.shape
    # fixed_noise = torch.randn(B, H, D, device=device).float()
    # fixed_time = torch.tensor([0.5] * B, device=device).float()

    with open(os.path.join(REPRODUCE_DIR, "batch_noise_time.pkl"), "rb") as f:
        batch_noise_time = dill.load(f)
        fixed_noise = torch.from_numpy(batch_noise_time['noise']).to(device)
        fixed_time = torch.from_numpy(batch_noise_time['time']).to(device)

    # # 4. 加载权重 (这是最难的一步，可能需要多次调试 key mapping)
    # # 假设你有一个从 JAX 导出的 params 字典 (通过 dill 或 msgpack 保存的)
    # # 如果没有，你需要修改 jax_infer.py 把 loaded_params_dict 保存下来
    # # 这里演示假设已经有了一个 params.pkl
    # # 如果没有文件，你可能需要先跑一下 jax_infer.py 并在那里保存 params
    #
    # # [临时方案]: 如果没有保存权重文件，这里只测试模型结构能否跑通
    # print("Skipping weight loading (run jax_infer.py to save 'params.pkl' first).")
    # print("Running forward pass with random weights for structure check...")

    #state_dict = torch.load("/data/PA/yejiaquan/pretrain_models/pi05_behavior_50t_pytorch/state.pth", map_location="cpu")
    state_dict = torch.load("final_b1k_pytorch.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    print("\n" + ">" * 20 + " DEBUG STEP 1: WEIGHT INSPECTION " + "<" * 20)
    patch_weight = model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight
    print(f"[PT] Patch Weight Shape: {patch_weight.shape}")
    # 预期: [1152, 3, 14, 14]
    # 打印一小块切片的值，用于和 JAX 原始权重（如果你能查到）对比
    # 或者单纯看数值分布是否正常
    print(f"[PT] Weight Slice [0,0,:3,:3]:\n{patch_weight[0, 0, :3, :3]}")
    print(f"[PT] Weight Mean: {patch_weight.mean().item():.6f}")

    # 5. 运行前向传播
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
    #with torch.no_grad():
        outputs = model.forward_detailed(
            pt_obs,
            pt_actions,
            noise=fixed_noise,
            time=fixed_time,
            num_flow_samples=1,  # for debug
            train=False,
        )

    print("-" * 40)
    print("PyTorch Forward Pass Successful!")
    print(f"Total Loss: {outputs['total_loss'].item():.6f}")
    print(f"Action Loss: {outputs['action_loss'].item():.6f}")
    
    print(f"action_loss_base_vel_x: {outputs['action_loss_base_vel_x'].item():.6f}")
    print(f"action_loss_base_vel_y: {outputs['action_loss_base_vel_y'].item():.6f}")
    print(f"action_loss_base_vel_z: {outputs['action_loss_base_vel_z'].item():.6f}")
    print(f"action_loss_trunk_0: {outputs['action_loss_trunk_0'].item():.6f}")
    print(f"action_loss_trunk_1: {outputs['action_loss_trunk_1'].item():.6f}")
    print(f"action_loss_trunk_2: {outputs['action_loss_trunk_2'].item():.6f}")
    print(f"action_loss_trunk_3: {outputs['action_loss_trunk_3'].item():.6f}")
    print(f"action_loss_left_arm_0: {outputs['action_loss_left_arm_0'].item():.6f}")
    print(f"action_loss_left_arm_1: {outputs['action_loss_left_arm_1'].item():.6f}")
    print(f"action_loss_left_arm_2: {outputs['action_loss_left_arm_2'].item():.6f}")
    print(f"action_loss_left_arm_3: {outputs['action_loss_left_arm_3'].item():.6f}")
    print(f"action_loss_left_arm_4: {outputs['action_loss_left_arm_4'].item():.6f}")
    print(f"action_loss_left_arm_5: {outputs['action_loss_left_arm_5'].item():.6f}")
    print(f"action_loss_left_arm_6: {outputs['action_loss_left_arm_6'].item():.6f}")
    print(f"action_loss_left_gripper: {outputs['action_loss_left_gripper'].item():.6f}")
    print(f"action_loss_right_arm_0: {outputs['action_loss_right_arm_0'].item():.6f}")
    print(f"action_loss_right_arm_1: {outputs['action_loss_right_arm_1'].item():.6f}")
    print(f"action_loss_right_arm_2: {outputs['action_loss_right_arm_2'].item():.6f}")
    print(f"action_loss_right_arm_3: {outputs['action_loss_right_arm_3'].item():.6f}")
    print(f"action_loss_right_arm_4: {outputs['action_loss_right_arm_4'].item():.6f}")
    print(f"action_loss_right_arm_5: {outputs['action_loss_right_arm_5'].item():.6f}")
    print(f"action_loss_right_arm_6: {outputs['action_loss_right_arm_6'].item():.6f}")
    print(f"action_loss_right_gripper: {outputs['action_loss_right_gripper'].item():.6f}")

    print(f"Fast Loss: {outputs.get('fast_loss', torch.tensor(0)).item():.6f}")
    print(f"Fast Accuracy: {outputs.get('fast_accuracy', torch.tensor(0)).item():.6f}")
    print(f"Subtask Accuracy: {outputs.get('subtask_accuracy', torch.tensor(0)).item():.6f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
