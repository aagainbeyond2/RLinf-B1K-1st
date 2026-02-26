import torch
import numpy as np
import re
import os
import dill

def convert_and_save_weights(pytorch_model, jax_params_flat, output_path="final_b1k_pytorch.pt"):
    """
    遍历 PyTorch 模型的所有参数，从 JAX 扁平化权重中按规则提取并填充。

    Args:
        pytorch_model: 初始化的 PyTorch 模型实例
        jax_params_flat: 扁平化的 JAX 权重字典 (Key: str, Value: numpy array)
                         key 示例: 'PaliGemma.llm.layers.mlp_1.gating_einsum.value'
        output_path: 保存路径
    """
    print(">>> 开始基于 PyTorch Key 的反向查找转换...")

    pt_state_dict = pytorch_model.state_dict()
    new_state_dict = {}

    # 辅助统计
    total_keys = len(pt_state_dict)
    matched_keys = 0

    for pt_key, pt_tensor in pt_state_dict.items():
        # print(f"Processing: {pt_key}") # 调试用
        #
        if pt_key != "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight":
            continue
        import pdb;pdb.set_trace()
        jax_data = None
        # ==============================================================================
        # 1. 视觉编码器 (Vision Tower) - SigLIP 深度修正版
        # ==============================================================================
        if "vision_tower" in pt_key:
            # 提取层号: encoder.layers.9.xxx
            layer_match = re.search(r"encoder\.layers\.(\d+)\.", pt_key)

            # --- 1.1 Transformer Layers ---
            if layer_match:
                layer_idx = int(layer_match.group(1))

                # --- A. Attention (Q/K/V/Out) ---
                if "self_attn" in pt_key:
                    base_jax = "PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0"

                    if "q_proj" in pt_key:   sub = "query"
                    elif "k_proj" in pt_key: sub = "key"
                    elif "v_proj" in pt_key: sub = "value"
                    elif "out_proj" in pt_key: sub = "out"

                    # 总是优先找 'kernel' (Linear weights), JAX list 里可能是 'weight'
                    # 根据你的 list: MultiHeadDotProductAttention_0.query.weight.value
                    jax_k = f"{base_jax}.{sub}.weight.value"

                    if jax_k in jax_params_flat:
                        # 1. 取出当前层 (Shape: [18432, 72] or similar)
                        raw_data = jax_params_flat[jax_k][layer_idx]
                        data = torch.from_numpy(np.array(raw_data))

                        if "bias" in pt_key:
                            # Bias 修复
                            # JAX Key Bias: [16, 72] (Heads, HeadDim) -> Flatten -> (1152)
                            # 你的 Key List 显示是 .bias.value
                            bias_key = jax_k.replace("weight", "bias")
                            if bias_key in jax_params_flat:
                                b_data = jax_params_flat[bias_key][layer_idx]
                                jax_data = torch.from_numpy(np.array(b_data)).flatten()

                        elif "out_proj" in pt_key:
                            # Output Proj (O)
                            # PyTorch: (Hidden, Hidden) = (1152, 1152)
                            # JAX: (1152, 72, 16) -> (Hidden, HeadDim, Heads) ? 需验证
                            # 或者是 (Hidden, Heads * HeadDim)

                            # 尝试 1: 如果是 (1152, 1152) -> 直接转置
                            if data.shape == (1152, 1152):
                                jax_data = data.t()

                            # 尝试 2: 如果是 (16, 82944) -> 你的报错显示了这个奇怪的形状
                            # 16 * 82944 = 1327104 = 1152 * 1152
                            # 这可能是 (Heads, HeadDim * Hidden) = (16, 72 * 1152)
                            elif data.shape == (16, 82944):
                                # Reshape -> (16, 72, 1152) (Heads, HeadDim, Hidden)
                                data = data.reshape(16, 72, 1152)
                                # Permute -> (1152, 16, 72) (Hidden, Heads, HeadDim) ?
                                # Out Proj Input is (Heads * HeadDim), Output is Hidden
                                # PT Weight is (Out, In) = (Hidden, Heads*HeadDim)
                                # 正确路径: (Heads, HeadDim, Hidden) -> Permute to (Hidden, Heads, HeadDim) -> Flatten
                                data = data.permute(2, 0, 1) # (1152, 16, 72)
                                jax_data = data.flatten(start_dim=1) # (1152, 1152)
                            elif data.shape == (16, 72, 1152):
                                data = data.permute(2, 0, 1) # (1152, 16, 72)
                                jax_data = data.flatten(start_dim=1) # (1152, 1152)
                            # # 尝试 3: 如果是 (1152, 72, 16)
                            # elif data.shape == (1152, 72, 16):
                            #     data = data.permute(0, 2, 1) # (1152, 16, 72)
                            #     jax_data = data.flatten(start_dim=1)

                        else:
                            # Input Proj (Q/K/V)
                            # PyTorch: (1152, 1152) (Heads*Dim, Hidden)
                            # JAX Raw: (18432, 72)

                            if data.shape == (18432, 72):
                                # 18432 = 16 * 1152 (Heads * Hidden)
                                # 意味着是 (Heads * Hidden, HeadDim)
                                # 还是 (Hidden * Heads, HeadDim)?

                                # 假设是 (Hidden, Heads, HeadDim) -> (1152, 16, 72)
                                # Reshape:
                                data = data.reshape(1152, 16, 72)
                                # PyTorch 需要 (Heads*HeadDim, Hidden)
                                # Permute to (Heads, HeadDim, Hidden)
                                data = data.permute(1, 2, 0) # (16, 72, 1152)
                                jax_data = data.reshape(1152, 1152)
                            elif data.shape == (1152, 16, 72):
                                data = data.permute(1, 2, 0)  # (16, 72, 1152)
                                jax_data = data.reshape(1152, 1152)
                            elif data.shape == (1152, 1152):
                                jax_data = data.t()

                # --- B. MLP ---
                elif "mlp" in pt_key:
                    base_jax = "PaliGemma.img.Transformer.encoderblock.MlpBlock_0"
                    if "fc1" in pt_key:   sub = "Dense_0"
                    elif "fc2" in pt_key: sub = "Dense_1"

                    suffix = "bias" if "bias" in pt_key else "weight"
                    jax_k = f"{base_jax}.{sub}.{suffix}.value"

                    if jax_k in jax_params_flat:
                        data = torch.from_numpy(np.array(jax_params_flat[jax_k][layer_idx]))
                        if "weight" in pt_key:
                            jax_data = data.t()
                        else:
                            jax_data = data

                # --- C. LayerNorm (Scale 修正) ---
                elif "layer_norm" in pt_key:
                    if "layer_norm1" in pt_key:   sub = "LayerNorm_0"
                    elif "layer_norm2" in pt_key: sub = "LayerNorm_1"

                    suffix = "bias" if "bias" in pt_key else "weight"

                    # 优先找 scale (JAX 标准)
                    scale_key = f"PaliGemma.img.Transformer.encoderblock.{sub}.scale.value"
                    weight_key = f"PaliGemma.img.Transformer.encoderblock.{sub}.weight.value"
                    bias_key = f"PaliGemma.img.Transformer.encoderblock.{sub}.bias.value"

                    target_jax_key = bias_key if suffix == "bias" else scale_key

                    # 如果找不到 scale，回退到 weight，但要检查形状
                    if suffix == "weight" and target_jax_key not in jax_params_flat:
                        target_jax_key = weight_key

                    if target_jax_key in jax_params_flat:
                        #data = jax_params_flat[target_jax_key][layer_idx]
                        if suffix == "weight":
                            data = jax_params_flat[target_jax_key][:, layer_idx]
                        else:
                            data = jax_params_flat[target_jax_key][layer_idx, :]

                        # 形状校验: 防止拿到 (27,) 这种 LayerScale 参数
                        # SigLIP LayerNorm dim 应该是 1152
                        if data.shape == (1152,):
                            jax_data = torch.from_numpy(np.array(data))
                        else:
                            print(f"⚠️ LayerNorm 形状异常 ({data.shape}), 可能是 LayerScale 参数，已跳过。PT 将使用默认初始化。")
                            # 这里直接返回 None，让 PyTorch 保持默认的 1 和 0

            # --- 1.2 Non-Layer Components (Embeddings) ---
            # 修正：Embeddings 在 JAX 里通常没有被 Stack，直接在顶层
            else:
                if "patch_embedding.weight" in pt_key:
                    # JAX: PaliGemma.img.embedding.weight.value
                    jax_data = jax_params_flat.get("PaliGemma.img.embedding.weight.value")
                    if jax_data is not None:
                        jax_data = torch.from_numpy(np.array(jax_data))
                        # Conv: (14, 14, 3, 1152) -> (1152, 3, 14, 14)
                        jax_data = jax_data.permute(3, 2, 0, 1)

                elif "patch_embedding.bias" in pt_key:
                    jax_data = jax_params_flat.get("PaliGemma.img.embedding.bias.value")

                elif "position_embedding.weight" in pt_key:
                    # JAX: PaliGemma.img.pos_embedding.value
                    jax_data = jax_params_flat.get("PaliGemma.img.pos_embedding.value")

                elif "post_layernorm" in pt_key:
                    suffix = "bias" if "bias" in pt_key else "weight"
                    # JAX: PaliGemma.img.Transformer.encoder_norm
                    # 这里同样优先找 scale
                    jax_name = "scale" if suffix == "weight" else "bias"
                    jax_k = f"PaliGemma.img.Transformer.encoder_norm.{jax_name}.value"

                    if jax_k not in jax_params_flat and suffix == "weight":
                        jax_k = f"PaliGemma.img.Transformer.encoder_norm.weight.value"

                    jax_data = jax_params_flat.get(jax_k)

        # ==============================================================================
        # 2. 语言模型 (LLM) - Gemma (Base & Expert)
        # ==============================================================================
        elif "language_model" in pt_key or "gemma_expert" in pt_key:
            # 判断是 Base 还是 Expert
            # 如果 pt_key 包含 'gemma_expert'，则对应 JAX key 带有后缀 _1
            is_expert = "gemma_expert" in pt_key
            suffix_idx = "_1" if is_expert else ""

            # 提取层号: layers.9.xxx
            layer_match = re.search(r"layers\.(\d+)\.", pt_key)

            if layer_match:
                layer_idx = int(layer_match.group(1))

                # --- 2.1 Attention (KV Split) ---
                if "self_attn" in pt_key:
                    # JAX: kv_einsum (融合), q_einsum, attn_vec_einsum (output)
                    # Key 格式: PaliGemma.llm.layers.attn.kv_einsum{_1}.w.value

                    if "q_proj.weight" in pt_key:
                        jax_k = f"PaliGemma.llm.layers.attn.q_einsum{suffix_idx}.w.value"
                        if jax_k in jax_params_flat:
                            jax_data = jax_params_flat[jax_k][layer_idx].T

                    elif "out_proj.weight" in pt_key or "o_proj.weight" in pt_key:
                        jax_k = f"PaliGemma.llm.layers.attn.attn_vec_einsum{suffix_idx}.w.value"
                        if jax_k in jax_params_flat:

                            raw_data = jax_params_flat[jax_k][layer_idx]
                            data = torch.from_numpy(np.array(raw_data))

                            if data.ndim == 3:
                                # JAX: [Heads, HeadDim, Out] -> [8, 256, 2048]
                                # PT Expects: [Out, In] -> [2048, 2048] (where In = Heads*HeadDim)
                                # 1. Permute to [Out, Heads, HeadDim] -> [2048, 8, 256]
                                # We need to move the last dim (Out) to front, and keep Heads/HeadDim together for flattening
                                data = data.permute(2, 0, 1)
                                # 2. Flatten the last two dims: [2048, 8*256] -> [2048, 2048]
                                jax_data = data.flatten(start_dim=1)
                            else:
                                jax_data = data.t()


                    elif "k_proj.weight" in pt_key or "v_proj.weight" in pt_key:
                        # 融合矩阵拆分
                        jax_k = f"PaliGemma.llm.layers.attn.kv_einsum{suffix_idx}.w.value"
                        if jax_k in jax_params_flat:
                            full_kv = jax_params_flat[jax_k][layer_idx] # (In, 2*Out)
                            # 拆分: dim 1 (feature dim)
                            # 你的观察：Shape (Layers, 2, ...) ?
                            # 如果你的 shape 是 (18, 2, ...), 那么应该取 [layer_idx][0] 或 [1]

                            if full_kv.ndim == 3 and full_kv.shape[0] == 2:
                                # Shape: (2, In, Head*Dim) 或者是 (2, In, Out)
                                k_part = full_kv[0]
                                v_part = full_kv[1]
                            else:
                                # Shape: (In, 2*Out) -> 传统 concat
                                split_size = full_kv.shape[-1] // 2
                                k_part, v_part = np.split(full_kv, 2, axis=-1)

                            if "k_proj" in pt_key: jax_data = k_part.T
                            if "v_proj" in pt_key: jax_data = v_part.T

                # --- 2.2 MLP (Gate/Up Split) ---
                elif "mlp" in pt_key:
                    # JAX: gating_einsum (融合 Gate+Up), linear (Down)
                    # Key: PaliGemma.llm.layers.mlp{_1}.gating_einsum.value

                    if "down_proj.weight" in pt_key:
                        jax_k = f"PaliGemma.llm.layers.mlp{suffix_idx}.linear.value"
                        # 注意: 有些版本 key 可能叫 linear.w.value，视你的 key list 而定
                        if jax_k not in jax_params_flat:
                             jax_k = f"PaliGemma.llm.layers.mlp{suffix_idx}.linear.w.value"

                        if jax_k in jax_params_flat:
                            jax_data = jax_params_flat[jax_k][layer_idx].T

                    elif "gate_proj.weight" in pt_key or "up_proj.weight" in pt_key:
                        jax_k = f"PaliGemma.llm.layers.mlp{suffix_idx}.gating_einsum.value"
                        if jax_k in jax_params_flat:
                            full_gate_up = jax_params_flat[jax_k][layer_idx]

                            # 根据你提供的 shape (18, 2, 1024, 4096)
                            # [Layer][Split][In][Out]
                            if full_gate_up.ndim == 3 and full_gate_up.shape[0] == 2:
                                gate_part = full_gate_up[0]
                                up_part   = full_gate_up[1]
                            else:
                                # 传统的 (In, 2*Out) 拆分
                                split_size = full_gate_up.shape[-1] // 2
                                gate_part, up_part = np.split(full_gate_up, 2, axis=-1)

                            if "gate_proj" in pt_key: jax_data = gate_part.T
                            if "up_proj" in pt_key:   jax_data = up_part.T

                # --- 2.3 Layer Norms ---
                # elif "layernorm" in pt_key:
                #     # input_layernorm -> pre_attention_norm
                #     # post_attention_layernorm -> pre_ffw_norm
                #     if "input_layernorm" in pt_key:   mid_name = "pre_attention_norm"
                #     elif "post_attention_layernorm" in pt_key: mid_name = "pre_ffw_norm"
                #
                #     jax_k = f"PaliGemma.llm.layers.{mid_name}{suffix_idx}.weight.value"
                #     if jax_k in jax_params_flat:
                #         #jax_data = jax_params_flat[jax_k][layer_idx]
                #         jax_data = jax_params_flat[jax_k][:,layer_idx]


                elif "layernorm" in pt_key:
                    # 1. 确定 JAX 的 Norm 名称
                    # input_layernorm -> pre_attention_norm
                    # post_attention_layernorm -> pre_ffw_norm
                    if "input_layernorm" in pt_key:
                        mid_name = "pre_attention_norm"
                    elif "post_attention_layernorm" in pt_key:
                        mid_name = "pre_ffw_norm"

                    # 2. 检查是否是 Adaptive Norm 里的 Dense 层
                    if "dense" in pt_key:
                        # PyTorch: input_layernorm.dense.weight
                        # JAX: pre_attention_norm_1.Dense_0.weight.value

                        # 确定 Dense 名称 (通常是 Dense_0)
                        sub_dense = "Dense_0"
                        suffix = "bias" if "bias" in pt_key else "weight"

                        jax_k = f"PaliGemma.llm.layers.{mid_name}{suffix_idx}.{sub_dense}.{suffix}.value"

                        if jax_k in jax_params_flat:
                            raw_data = jax_params_flat[jax_k][layer_idx]
                            data = torch.from_numpy(np.array(raw_data))

                            # Linear 层权重需要转置
                            if suffix == "weight":
                                jax_data = data.t()
                            else:
                                jax_data = data

                    else:
                        # 3. 标准 LayerNorm (Scale/Weight)
                        suffix = "bias" if "bias" in pt_key else "weight"

                        # JAX Key 构造
                        jax_k = f"PaliGemma.llm.layers.{mid_name}{suffix_idx}.{suffix}.value"

                        # 如果是 weight，JAX 可能叫 scale (也可能叫 weight)
                        if suffix == "weight" and jax_k not in jax_params_flat:
                            # 尝试找 .scale
                            jax_k_scale = f"PaliGemma.llm.layers.{mid_name}{suffix_idx}.scale.value"
                            if jax_k_scale in jax_params_flat:
                                jax_k = jax_k_scale

                        if jax_k in jax_params_flat:
                            #jax_data = jax_params_flat[jax_k][layer_idx]
                            jax_data = jax_params_flat[jax_k][:, layer_idx]




            # --- 2.4 LLM Non-Layer (Embed, Final Norm) ---
            else:
                if "embed_tokens" in pt_key:
                    # 注意: Expert 可能共享 Embedding，也可能不共享
                    # 通常 Base 和 Expert 共享 input_embedding
                    jax_data = jax_params_flat.get("PaliGemma.llm.embedder.input_embedding.value")
                elif "norm.weight" in pt_key:
                    # final_norm
                    jax_k = f"PaliGemma.llm.final_norm{suffix_idx}.weight.value"
                    jax_data = jax_params_flat.get(jax_k)


        # ==============================================================================
        # 3. Action Expert Components (Top Level)
        # ==============================================================================
        else:
            # 1. 构造可能的 JAX Key 列表
            # JAX nnx.Linear 使用 'kernel'，PyTorch 使用 'weight'
            candidates = [
                pt_key + ".value",  # 尝试直接加 .value
                pt_key.replace("weight", "kernel") + ".value"  # 尝试把 weight 换成 kernel
            ]

            # 特殊情况：Task Embeddings
            if "embeddings.weight" in pt_key:
                candidates.append(pt_key.replace(".weight", ".embedding.value"))

            # 2. 查找存在的 Key
            jax_data = None
            for jax_k in candidates:
                if jax_k in jax_params_flat:
                    # 找到了！
                    data = jax_params_flat[jax_k]
                    # 转为 Tensor
                    if isinstance(data, np.ndarray):
                        data = torch.from_numpy(data)

                    # 3. 判断是否需要转置
                    # 如果是 Linear 层的权重 (二维矩阵)，必须转置！
                    # gate_sincos, gate_task, fusion_layer 等都是 Linear
                    if "weight" in pt_key and data.ndim == 2 and "embedding" not in pt_key:
                        # 获取 PyTorch 目标形状
                        pt_shape = pt_tensor.shape  # e.g., (1024, 4096)
                        jax_shape = data.shape  # e.g., (1024, 4096) or (4096, 1024)

                        # ====================================================
                        # [智能修正] 基于形状对比决定是否转置
                        # ====================================================
                        if jax_shape == pt_shape:
                            print(f"✅ Shape matches exactly for {pt_key}: {jax_shape}. Keep as is.")
                            # 形状完美匹配 (Out, In)，直接用！
                            jax_data = data

                        elif jax_shape == (pt_shape[1], pt_shape[0]):
                            print(f"?? Transposing {pt_key}: {jax_shape} -> {pt_shape}")
                            # 形状是转置关系 (In, Out)，需要转置！
                            jax_data = data.t()

                        else:
                            print(f"⚠️ Shape mismatch for {pt_key}: PT{pt_shape} vs JAX{jax_shape}. Check manually!")
                            # 既不匹配也不互为转置，可能是 Reshape 问题，保留原样交给后续兜底
                            jax_data = data
                    else:
                        # Bias 或其他参数
                        jax_data = data

                    break  # 找到后停止尝试其他 key

            # 4. 尝试处理 KV Transform (系数不需要转置)
            if jax_data is None:
                if pt_key + ".value" in jax_params_flat:
                    jax_data = torch.from_numpy(jax_params_flat[pt_key + ".value"])

        # ==============================================================================
        # 4. Multi-Modal Projector (Vision -> LLM)
        # ==============================================================================
        if "multi_modal_projector" in pt_key:
            # PyTorch: ...multi_modal_projector.linear.weight
            # JAX: PaliGemma.img.head.weight.value

            suffix = "bias" if "bias" in pt_key else "weight"
            jax_k = f"PaliGemma.img.head.{suffix}.value"
            if jax_k in jax_params_flat:
                data = torch.from_numpy(np.array(jax_params_flat[jax_k]))
                # Linear 层需要转置
                # if suffix == "weight":
                #     jax_data = data.t()
                # else:
                #     jax_data = data
                jax_data = data  # 本就是 (2048, 512)


        # ==============================================================================
        # 5. LM Head (Weight Tying)
        # ==============================================================================
        elif "lm_head.weight" in pt_key:
            if "gemma_expert" in pt_key:
                # 情况 A: Expert 模型 (gemma_expert)
                # Expert 使用 action_out_proj (1024->32)，不使用 lm_head (1024->257152)。
                # PyTorch 的 GemmaForCausalLM 自动创建了 lm_head，但它是多余的。
                # JAX Checkpoint 里也没有对应的权重。
                # -> 赋予全0，不报错。
                print(f"ℹ️√ Death weight use all zeros unused Expert lm_head: {pt_key}")
                jax_data = torch.zeros_like(pt_tensor)
            else:
                # 情况 B: Base 模型 (paligemma)
                # PyTorch: lm_head.weight
                # JAX: PaliGemma.llm.embedder.input_embedding.value
                jax_k = "PaliGemma.llm.embedder.input_embedding.value"

                if jax_k in jax_params_flat:
                    raw_data = jax_params_flat[jax_k]
                    jax_data = torch.from_numpy(np.array(raw_data))

                    if jax_data.shape != pt_tensor.shape:
                        print(f"⚠️ Base Model lm_head mismatch! PT {pt_tensor.shape} vs JAX {jax_data.shape}")
                        continue

        # ==============================================================================
        # 6. Expert Final Norm (Adaptive / Dense)
        # ==============================================================================
        elif "gemma_expert.model.norm" in pt_key:
            # PyTorch: ...gemma_expert.model.norm.dense.weight
            # JAX: PaliGemma.llm.final_norm_1.Dense_0.weight.value

            # 1. 确定后缀 (_1 因为是 Expert)
            suffix_idx = "_1"

            # 2. 检查是否是 Dense 结构
            if "dense" in pt_key:
                sub_dense = "Dense_0"
                suffix = "bias" if "bias" in pt_key else "weight"

                jax_k = f"PaliGemma.llm.final_norm{suffix_idx}.{sub_dense}.{suffix}.value"

                if jax_k in jax_params_flat:
                    data = torch.from_numpy(np.array(jax_params_flat[jax_k]))
                    if suffix == "weight":
                        jax_data = data.t()
                    else:
                        jax_data = data
            else:
                # 普通 Norm (兜底)
                suffix = "bias" if "bias" in pt_key else "weight"
                jax_k = f"PaliGemma.llm.final_norm{suffix_idx}.{suffix}.value"
                if jax_k in jax_params_flat:
                    jax_data = torch.from_numpy(np.array(jax_params_flat[jax_k]))

        # --- [新增] 专用修复: Fast Token Embedding ---
        elif "fast_token_embedding.weight" in pt_key:
            # PT: fast_token_embedding.weight
            # JAX: fast_token_embedding.embedding.value
            jax_k = "fast_token_embedding.embedding.value"
            if jax_k in jax_params_flat:
                jax_data = torch.from_numpy(np.array(jax_params_flat[jax_k]))
                # Embedding 层不需要转置

        # ==============================================================================
        # 4. 赋值与保存
        # ==============================================================================
        if jax_data is not None:
            # 类型转换 numpy -> torch
            if isinstance(jax_data, np.ndarray):
                jax_data = torch.from_numpy(jax_data)

            # 形状检查
            if jax_data.shape != pt_tensor.shape:
                # 尝试自动修复 1D vs 0D 问题
                jax_data_old_shape = jax_data.shape
                if jax_data.numel() == pt_tensor.numel():
                    jax_data = jax_data.reshape(pt_tensor.shape)
                    print(f"⚠️ √   Shape Mismatch for {pt_key}: PT {pt_tensor.shape} vs JAX {jax_data_old_shape}, force reshape ...")
                else:
                    print(f"⚠️ Shape Mismatch for {pt_key}: PT {pt_tensor.shape} vs JAX {jax_data_old_shape}")
                    continue # 跳过无法匹配的

            # 填充权重
            new_state_dict[pt_key] = jax_data
            matched_keys += 1
            #print(f"√ Shape Match for {pt_key}: PT {pt_tensor.shape} vs JAX {jax_data.shape}")
        else:
            # 打印未找到的 Key 以便调试 (忽略一些统计量)
            if "num_batches_tracked" not in pt_key and "running_" not in pt_key:
                print(f"❌ Missing JAX source for: {pt_key}")

    print(f"\n>>> 转换完成. 匹配进度: {matched_keys}/{total_keys}")

    # 保存
    torch.save(new_state_dict, output_path)
    print(f">>> PyTorch state_dict 已保存至: {output_path}")

# ==============================================================================
# 使用入口
# ==============================================================================
if __name__ == "__main__":
    # ================= 配置区域 =================
    # 1. JAX 权重文件夹路径
    JAX_CKPT_PATH = "/root/.cache/openpi/openpi-assets/checkpoints/pi05_behavior_50t/params/snapshots/114902410a6439ad1a0b58a112b84159d3ab3f65/params"

    # 2. 输出路径
    INPUT_PATH = "/data/PA/yejiaquan/pretrain_models/pi05_behavior_50t_pytorch/state.pth"
    # ===========================================


    # 1. 假设你已经用之前的脚本加载了 JAX 权重 (jax_params)
    jax_params_flat = torch.load(INPUT_PATH, map_location="cpu")

    # 2. 初始化你的 PyTorch 模型
    import sys
    sys.path.append("/ms/PA/yejiaquan/code/behavior-1k-solution-main/openpi/src/openpi/models_pytorch")
    from pi_behavior_pytorch import PiBehaviorPytorch
    REPRODUCE_DIR = "/ms/PA/yejiaquan/b1k_models_pytorch/reproduce/"

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

    # 3. 运行
    convert_and_save_weights(model, jax_params_flat)
    pass