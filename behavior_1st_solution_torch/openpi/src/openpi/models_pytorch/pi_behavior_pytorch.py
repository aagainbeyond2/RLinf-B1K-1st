import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from typing import Optional, Tuple, List, Dict, Any
from transformers.cache_utils import DynamicCache

# 假设这些配置在外部文件中定义，这里为了代码完整性进行模拟
# 在实际使用中，请替换为 b1k.models.pi_behavior_config 中的真实数据
TASK_NUM_STAGES = [10] * 100  # 示例数据
TASK_STAGE_OFFSETS = [0] * 100 # 示例数据
TOTAL_TASK_STAGE_EMBEDDINGS = 596
MAX_NUM_STAGES = 15

from b1k.models.pi_behavior_config import (
    TASK_NUM_STAGES,
    MAX_NUM_STAGES,
    TOTAL_TASK_STAGE_EMBEDDINGS,
    TASK_STAGE_OFFSETS
)


from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    get_safe_dtype
)
# 引用你提供的 gemma_pytorch 中的模型
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

class KVCacheTransform(nn.Module):
    """
    Transforms prefix KV cache by mixing across layers.
    Corresponds to JAX 'KVCacheTransform'.
    """
    def __init__(self, num_layers: int, head_dim: int, num_kv_heads: int):
        super().__init__()
        # K transformation: [dest_layer, src_layer]
        self.k_coeffs = nn.Parameter(torch.eye(num_layers))
        # K bias: [layer, num_kv_heads, head_dim]
        self.k_bias = nn.Parameter(torch.zeros(num_layers, num_kv_heads, head_dim))

        # V transformation
        self.v_coeffs = nn.Parameter(torch.eye(num_layers))
        self.v_bias = nn.Parameter(torch.zeros(num_layers, num_kv_heads, head_dim))

    def forward(self, past_key_values: Any):
        """
        Args:
            past_key_values: List of length num_layers, each tuple is (K, V).
                             K/V shape in HF: [batch, num_kv_heads, seq_len, head_dim]
        """
        cache_obj = None
        if (
            past_key_values is not None
            and hasattr(past_key_values, "get_seq_length")
            and hasattr(past_key_values, "key_cache")
            and hasattr(past_key_values, "value_cache")
        ):
            cache_obj = past_key_values
            past_key_values = list(zip(cache_obj.key_cache, cache_obj.value_cache))
        # Stack layers: [layers, batch, heads, seq, dim]
        # Note: HF PyTorch cache is typically (batch, heads, seq, dim)

        # 1. Unpack and Stack
        ks = torch.stack([layer[0] for layer in past_key_values], dim=0)
        vs = torch.stack([layer[1] for layer in past_key_values], dim=0)

        # ks shape: [layers, batch, heads, seq, dim]
        target_dtype = ks.dtype
        if self.k_coeffs.dtype != target_dtype:
            k_coeffs = self.k_coeffs.to(dtype=target_dtype)
        else:
            k_coeffs = self.k_coeffs
        if self.v_coeffs.dtype != target_dtype:
            v_coeffs = self.v_coeffs.to(dtype=target_dtype)
        else:
            v_coeffs = self.v_coeffs
        if self.k_bias.dtype != target_dtype:
            k_bias = self.k_bias.to(dtype=target_dtype)
        else:
            k_bias = self.k_bias
        if self.v_bias.dtype != target_dtype:
            v_bias = self.v_bias.to(dtype=target_dtype)
        else:
            v_bias = self.v_bias

        # 2. Transform K
        # Einsum: [dest, src] @ [src, batch, heads, seq, dim] -> [dest, batch, heads, seq, dim]
        # JAX shape was [layers, batch, seq, heads, dim], logic matches but dims permuted
        k_new = torch.einsum('os,sbhld->obhld', k_coeffs, ks)

        # Add bias: [layers, 1, heads, 1, dim]
        k_bias_expanded = k_bias.unsqueeze(1).unsqueeze(3) # [L, 1, H, 1, D]
        k_new = k_new + k_bias_expanded

        # 3. Transform V
        v_new = torch.einsum('os,sbhld->obhld', v_coeffs, vs)
        v_bias_expanded = v_bias.unsqueeze(1).unsqueeze(3)
        v_new = v_new + v_bias_expanded

        new_past_key_values = [(k_new[i], v_new[i]) for i in range(k_new.shape[0])]

        if cache_obj is not None:
            new_keys = [kv[0] for kv in new_past_key_values]
            new_vals = [kv[1] for kv in new_past_key_values]
            if isinstance(cache_obj.key_cache, list):
                cache_obj.key_cache[:] = new_keys
            else:
                cache_obj.key_cache = new_keys
            if isinstance(cache_obj.value_cache, list):
                cache_obj.value_cache[:] = new_vals
            else:
                cache_obj.value_cache = new_vals
            return cache_obj

        return new_past_key_values

class PiBehaviorPytorch(PI0Pytorch):
    def __init__(self, config):
        # Initialize parent PI0Pytorch (which sets up PaliGemmaWithExpertModel)
        super().__init__(config, is_pi05=True)

        self.config = config

        # --- Config Parameters ---
        self.task_embedding_dim = config.task_embedding_dim if hasattr(config, 'task_embedding_dim') else 2048
        self.subtask_encoding_dim = self.task_embedding_dim // 2
        self.num_tasks = config.num_tasks

        # --- Task & Stage Embeddings ---
        self.task_embeddings = nn.Embedding(self.num_tasks, self.task_embedding_dim)

        # Stage predictor (from VLM width to MAX_NUM_STAGES)
        # Note: paligemma_config.width is usually 2048 or similar
        self.stage_pred_from_vlm = nn.Linear(self.paligemma_with_expert.paligemma.config.text_config.hidden_size, MAX_NUM_STAGES)

        self.task_stage_embeddings = nn.Embedding(TOTAL_TASK_STAGE_EMBEDDINGS, self.subtask_encoding_dim)

        # --- Fusion Layers ---
        fusion_input_dim = self.task_embedding_dim + 2 * self.subtask_encoding_dim # 2048 + 1024 + 1024 = 4096

        self.gate_sincos = nn.Linear(fusion_input_dim, self.subtask_encoding_dim)
        self.gate_task_stage = nn.Linear(fusion_input_dim, self.subtask_encoding_dim)
        self.gate_task = nn.Linear(fusion_input_dim, self.task_embedding_dim)

        self.fusion_layer1 = nn.Linear(fusion_input_dim, self.task_embedding_dim * 2)
        self.fusion_layer2 = nn.Linear(self.task_embedding_dim * 2, self.task_embedding_dim)

        self.stage_projection = nn.Linear(2 * self.subtask_encoding_dim, self.task_embedding_dim)

        # --- KV Cache Transform ---
        if config.use_kv_transform:
            vlm_config = self.paligemma_with_expert.paligemma.config.text_config
            self.kv_transform = KVCacheTransform(
                num_layers=vlm_config.num_hidden_layers,
                head_dim=vlm_config.head_dim,
                num_kv_heads=vlm_config.num_key_value_heads
            )
        else:
            self.kv_transform = None

        # --- Aux Components ---
        # FAST auxiliary components (Optional)
        if hasattr(config, 'use_fast_auxiliary') and config.use_fast_auxiliary:
            self.fast_token_embedding = nn.Embedding(config.fast_vocab_size, self.task_embedding_dim)
            self.fast_token_proj = nn.Linear(self.task_embedding_dim, config.fast_vocab_size)

        # --- Correlated Noise Components ---
        self.use_correlated_noise = getattr(config, 'use_correlated_noise', False)
        self.correlation_beta = getattr(config, 'correlation_beta', 0.95) # 默认收缩系数

        # 注册一个 Buffer 来存储 Cholesky 矩阵
        # persistent=False 表示它可能不包含在 state_dict 中 (视需求而定，JAX版是不存checkpoint的)
        # 形状为 [Flat_Action_Dim, Flat_Action_Dim]
        flat_dim = self.config.action_horizon * self.config.action_dim
        self.register_buffer(
            "action_correlation_cholesky",
            torch.eye(flat_dim),
            persistent=False
        )
        self.correlation_loaded = False

    def encode_subtask_state(self, subtask_state: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode subtask state using cos/sin positional encoding.
        Corresponds to JAX `encode_subtask_state`.
        """
        device = subtask_state.device

        # Convert constants to tensor on device
        task_num_stages = torch.tensor(TASK_NUM_STAGES, device=device)[task_ids]

        # Normalize: [0, 1]
        normalized_state = subtask_state.float() / torch.clamp(task_num_stages.float() - 1.0, min=1.0)

        # Use existing sinusoidal function, flattened for the batch
        # Note: create_sinusoidal_pos_embedding expects [Batch], returns [Batch, Dim]
        return create_sinusoidal_pos_embedding(
            normalized_state,
            self.subtask_encoding_dim,
            min_period=1e-3,
            max_period=1.0,
            device=device
        )

    def fuse_task_and_subtask(self, task_embedding, task_ids, subtask_state):
        """
        Fuse task embedding with subtask state.
        Returns [B, 4, D] tensor with different fusion strategies.
        """
        batch_size = task_embedding.shape[0]

        # 1. SinCos Encoding
        sincos_encoding = self.encode_subtask_state(subtask_state, task_ids) # [B, 1024]
        sincos_encoding = sincos_encoding.to(dtype=task_embedding.dtype)

        # 2. Task-Specific Stage Embedding
        task_stage_offsets = torch.tensor(TASK_STAGE_OFFSETS, device=task_embedding.device)[task_ids]
        task_stage_idx = task_stage_offsets + subtask_state
        task_stage_emb = self.task_stage_embeddings(task_stage_idx) # [B, 1024]

        # Concatenate inputs
        #all_inputs = torch.cat([task_embedding, sincos_encoding, task_stage_emb], dim=-1).float() # [B, 4096]
        all_inputs = torch.cat([task_embedding, sincos_encoding, task_stage_emb], dim=-1) # [B, 4096]
        all_inputs = all_inputs.to(self.gate_sincos.weight.dtype)

        # Gates
        gate_sincos = torch.sigmoid(self.gate_sincos(all_inputs))
        gate_task_stage = torch.sigmoid(self.gate_task_stage(all_inputs))
        gate_task = torch.sigmoid(self.gate_task(all_inputs))

        # Strategy 1: Task Gated
        task_gated = task_embedding * gate_task

        # Strategy 2: Balanced Fusion
        x = F.relu(self.fusion_layer1(all_inputs))
        balanced_fusion = self.fusion_layer2(x)

        # Strategy 3: Stage Dominant
        # gated_stage_features = torch.cat([
        #     sincos_encoding * gate_sincos,
        #     task_stage_emb * gate_task_stage
        # ], dim=-1).float()
        gated_stage_features = torch.cat([
            sincos_encoding * gate_sincos,
            task_stage_emb * gate_task_stage
        ], dim=-1)
        gated_stage_features = gated_stage_features.to(self.stage_projection.weight.dtype)

        stage_dominant = self.stage_projection(gated_stage_features)

        # Strategy 4: Pure Stage
        #pure_stage = torch.cat([sincos_encoding, task_stage_emb], dim=-1).float()
        pure_stage = torch.cat([sincos_encoding, task_stage_emb], dim=-1)
        pure_stage = pure_stage.to(dtype=task_embedding.dtype)

        # Stack: [B, 4, 2048]
        #fused_embeddings = torch.stack([task_gated, balanced_fusion, stage_dominant, pure_stage], dim=1).float()
        fused_embeddings = torch.stack([task_gated, balanced_fusion, stage_dominant, pure_stage], dim=1)
        fused_embeddings = fused_embeddings.to(dtype=task_embedding.dtype)

        return fused_embeddings

    def embed_prefix(
            self,
            images: List[torch.Tensor],
            img_masks: List[torch.Tensor],
            tokenized_prompt: torch.Tensor,
            state: torch.Tensor,
            fast_tokens: Optional[torch.Tensor] = None,
            fast_token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed prefix: images + task + state + FAST_tokens.
        [修复]: 显式支持 5D [Batch, Seq, C, H, W] 输入，自动展平处理。
        """
        embs = []
        pad_masks = []
        att_masks = []

        # 1. Images (Vision Tower)
        # -----------------------------------------------
        for img, img_mask in zip(images, img_masks):
            # img shape: [B, S, C, H, W] (5D) or [B, C, H, W] (4D)

            # --- [核心修复: 如果是 5D 视频，先展平再编码] ---
            if img.ndim == 5:
                b, s, c, h, w = img.shape
                # 1. Fold: 合并 Batch 和 Seq -> [B*S, C, H, W] (变成标准 4D 图片)
                img_flat = img.view(b * s, c, h, w)

                # 2. Encode: Vision Tower 处理 4D 图片
                # Output: [B*S, Num_Patches, Dim]
                # 使用 checkpointing 以节省显存 (和原逻辑保持一致)
                img_emb_flat = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img_flat)

                # 3. Unfold: [B*S, N, D] -> [B, S, N, D]
                _, num_patches, emb_dim = img_emb_flat.shape
                img_emb = img_emb_flat.view(b, s, num_patches, emb_dim)

                # 4. Flatten Sequence: [B, S, N, D] -> [B, S*N, D]
                # 将“时间步”和“Patch”两个维度合并，作为 LLM 的长序列输入
                img_emb = img_emb.flatten(1, 2)

                # 5. Handle Mask:
                # img_mask 是 [B, S], 需要扩展到每个 Patch 上
                # [B, S] -> [B, S, 1] -> [B, S, N] -> [B, S*N]
                if img_mask.ndim == 1:
                    img_mask = img_mask[:, None].expand(b, s)
                mask_expanded = img_mask.unsqueeze(-1).expand(b, s, num_patches)
                mask_flat = mask_expanded.flatten(1, 2)  # [B, Total_Img_Tokens]
                embs.append(img_emb)
                pad_masks.append(mask_flat)

                # Attention Mask ID (0 for all prefix tokens)
                att_masks += [0] * img_emb.shape[1]

            else:
                # --- [标准 4D 处理 (保留原有逻辑以兼容单帧输入)] ---
                img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs


            # print(f"[PT] img_embedding_weight Mean: {self.paligemma_with_expert.state_dict()['paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight'].mean().item():.6f}")
            # print(f"[PT] img_embedding_weight Std: {self.paligemma_with_expert.state_dict()['paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight'].std().item():.6f}")
            # print(f"[PT] img_embedding_weight Sum: {self.paligemma_with_expert.state_dict()['paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight'].sum().item():.6f}")
            # print(f"[PT] img_embedding_bias Mean: {self.paligemma_with_expert.state_dict()['paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias'].mean().item():.6f}")
            # print(f"[PT] img_embedding_bias Std: {self.paligemma_with_expert.state_dict()['paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias'].std().item():.6f}")
            # print(f"[PT] img_embedding_bias Sum: {self.paligemma_with_expert.state_dict()['paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias'].sum().item():.6f}")
            # print(f"[PT] img Mean: {img.mean().item():.6f}")
            # print(f"[PT] img_emb Mean: {img_emb.mean().item():.6f}")
            # print(f"[PT] img_emb Std: {img_emb.std().item():.6f}")
            # print(f"[PT] img_emb Sum: {img_emb.sum().item():.6f}")

        # 2. Task + Subtask Fusion
        # -----------------------------------------------
        task_ids = tokenized_prompt[:, 0]
        subtask_state = tokenized_prompt[:, 1]

        base_task_embedding = self.task_embeddings(task_ids)
        fused_task_embeddings = self.fuse_task_and_subtask(base_task_embedding, task_ids, subtask_state)

        # [B, 5, 2048]
        task_sequence = torch.cat([base_task_embedding.unsqueeze(1), fused_task_embeddings], dim=1)
        #task_sequence = task_sequence * math.sqrt(task_sequence.shape[-1])

        embs.append(task_sequence)

        # print(f"[PT] task_sequence Mean: {task_sequence.mean().item():.6f}")
        # print(f"[PT] task_sequence Std: {task_sequence.std().item():.6f}")
        # print(f"[PT] task_sequence Sum: {task_sequence.sum().item():.6f}")

        bsize = task_sequence.shape[0]
        task_mask = torch.ones(bsize, 5, dtype=torch.bool, device=task_sequence.device)
        pad_masks.append(task_mask)
        #att_masks += [0] * 5
        att_masks += [0] + [1, 0, 0, 0]

        # 3. Discretized State
        # -----------------------------------------------
        bins = torch.linspace(-1, 1, 256 + 1, device=state.device)[:-1]
        eps = 1e-6  # add epsilon to align 0.0 boundary with JAX
        discretized_state = torch.bucketize(state + eps, bins) - 1
        discretized_state = torch.clamp(discretized_state, 0, 255)

        embed_tokens = self.paligemma_with_expert.paligemma.language_model.embed_tokens
        state_embs = embed_tokens(discretized_state)
        state_embs = state_embs * math.sqrt(state_embs.shape[-1])

        embs.append(state_embs)

        # print(f"[PT] state_embs Mean: {state_embs.mean().item():.6f}")
        # print(f"[PT] state_embs Std: {state_embs.std().item():.6f}")
        # print(f"[PT] state_embs Sum: {state_embs.sum().item():.6f}")

        pad_masks.append(torch.ones(bsize, state_embs.shape[1], dtype=torch.bool, device=state.device))
        att_masks += [0] * state_embs.shape[1]

        # 4. FAST Tokens
        # -----------------------------------------------
        if self.config.use_fast_auxiliary and fast_tokens is not None:
            bos_token = torch.zeros(bsize, 1, dtype=torch.long, device=fast_tokens.device)
            shifted_tokens = torch.cat([bos_token, fast_tokens[:, :-1]], dim=1)

            fast_emb = self.fast_token_embedding(shifted_tokens)
            embs.append(fast_emb)

            # print(f"[PT] fast_emb Mean: {fast_emb.mean().item():.6f}")
            # print(f"[PT] fast_emb Std: {fast_emb.std().item():.6f}")
            # print(f"[PT] fast_emb Sum: {fast_emb.sum().item():.6f}")

            bos_mask = torch.ones(bsize, 1, dtype=torch.bool, device=fast_tokens.device)
            shifted_mask = torch.cat([bos_mask, fast_token_mask[:, :-1]], dim=1)
            pad_masks.append(shifted_mask)
            att_masks += [1] * shifted_tokens.shape[1]

        # Concatenate all
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # Construct att_masks tensor
        att_masks = torch.tensor(att_masks, dtype=torch.long, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def sample_time(self, batch_size, device):
            """
            Sample time steps from Beta distribution.
            Matches JAX: jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
            """
            # PyTorch doesn't have direct beta sampler in torch.xyz, use distributions
            m = torch.distributions.beta.Beta(torch.tensor([1.5], device=device), torch.tensor([1.0], device=device))
            t = m.sample((batch_size,)).squeeze(-1)
            t = t * 0.999 + 0.001
            return t

    def _preprocess_observation(self, observation, *, train=True):
        """
        重写基类方法以适配 5D 视频输入 [Batch, Seq, Channel, Height, Width]。

        逻辑：
        1. 检测并展平 5D 张量 -> 4D (Batch * Seq)。
        2. 调用基类同款的标准预处理 (Resize, Normalize)。
        3. 恢复 5D 形状。
        4. 返回基类要求的 tuple 格式。
        """
        # 局部导入，防止循环依赖
        from openpi.models_pytorch import preprocessing_pytorch as _preprocessing

        # 1. 记录原始形状并执行 Fold (展平)
        # ---------------------------------------------------------
        original_shapes = {}

        # 处理 Images
        for name, img in observation.images.items():
            if img.ndim == 5:
                b, s, c, h, w = img.shape
                original_shapes[name] = (b, s)
                # [B, S, C, H, W] -> [B*S, C, H, W]
                observation.images[name] = img.reshape(b * s, c, h, w)

        # 处理 Masks (如果存在且是 2D [B, S])
        for name, mask in observation.image_masks.items():
            if mask.ndim == 2:
                b, s = mask.shape
                # 记录一下 shape (通常和 image 一致，这里为了安全)
                # [B, S] -> [B*S]
                observation.image_masks[name] = mask.reshape(b * s)

        # 2. 调用标准预处理
        # ---------------------------------------------------------
        # 此时所有 tensor 都是标准的 4D (Image) 或 1D (Mask)，预处理函数可以正常工作
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)

        # 3. 恢复形状 (Unfold)
        # ---------------------------------------------------------
        for name, (b, s) in original_shapes.items():
            # 恢复 Image
            if name in observation.images:
                img = observation.images[name]
                # 此时 img 是经过 resize 后的 [B*S, C, H_new, W_new]
                if img.ndim == 4:
                    _, c, h, w = img.shape
                    observation.images[name] = img.reshape(b, s, c, h, w)

            # 恢复 Mask
            if name in observation.image_masks:
                mask = observation.image_masks[name]
                # 此时 mask 是 [B*S]
                if mask.ndim == 1:
                    observation.image_masks[name] = mask.reshape(b, s)

        # 4. 返回基类要求的标准格式
        # ---------------------------------------------------------
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def forward_detailed(
        self,
        observation,
        actions,
        noise=None,
        time=None,
        num_flow_samples: int = 1,
        train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Comprehensive forward pass computing losses, KV mixing, etc.
        Matches JAX compute_detailed_loss logic including FAST aux and multi-sample flow matching.
        """
        losses = {}

        # Preprocess
        #images, img_masks, _, _, state = self._preprocess_observation(observation, train=True)
        images, img_masks, _, _, state = self._preprocess_observation(observation, train=train)
        tokenized_prompt = observation.tokenized_prompt # Has task info

        # Optional FAST tokens
        fast_tokens = getattr(observation, 'fast_tokens', None)
        fast_token_mask = getattr(observation, 'fast_token_mask', None)

        # -------------------------------------------------------------------------
        # 1. Embed Prefix (Images + Task + State + [FAST])
        # -------------------------------------------------------------------------
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokenized_prompt, state, fast_tokens, fast_token_mask
        )

        # # [Debug A]
        # print("\n" + "=" * 30 + " DEBUG A: EMBEDDINGS " + "=" * 30)
        # print(f"[PT] prefix_embs  shape: {prefix_embs.shape}")
        # print(f"[PT] prefix_embs  mean:  {prefix_embs.mean().item():.6f}")
        # print(f"[PT] prefix_embs  std:   {prefix_embs.std().item():.6f}")
        # print(f"[PT] prefix_embs  sum:   {prefix_embs.sum().item():.6f}")
        # print(f"[PT] images[0]  mean:   {images[0].mean().item():.6f}")
        # print(f"[PT] images[1]  mean:   {images[1].mean().item():.6f}")
        # print(f"[PT] images[2]  mean:   {images[2].mean().item():.6f}")

        # Prepare VLM masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # -------------------------------------------------------------------------
        # 2. Run VLM Part (Prefix Only) -> Get KV Cache & Hidden States
        # -------------------------------------------------------------------------
        # Force eager attention for cache access availability
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos_ids,
            inputs_embeds=[prefix_embs, None],
            use_cache=True
        )

        # -------------------------------------------------------------------------
        # 3. Stage Prediction Loss & FAST Loss
        # -------------------------------------------------------------------------
        # A. Stage Prediction
        # Assuming fixed structure: Images -> Task. Base Task is at index `num_image_tokens`
        # SigLIP 224x224 patch14 -> 256 tokens per image
        # total_image_frames = sum([img.shape[1] for img in images]) # = 3
        tokens_per_image = 256
        total_image_frames = 0
        for img in images:
            if img.ndim==4:
                # training
                total_image_frames += 1
            elif img.ndim==5:
                # eval
                total_image_frames += img.shape[1]
        base_task_token_idx = total_image_frames * tokens_per_image # = 768

        # Mask invalid stages logic for loss calculation
        base_task_output = prefix_out[:, base_task_token_idx, :]
        if base_task_output.dtype != self.stage_pred_from_vlm.weight.dtype:
            base_task_output = base_task_output.to(dtype=self.stage_pred_from_vlm.weight.dtype)
        subtask_logits = self.stage_pred_from_vlm(base_task_output) # [B, 15]
        task_ids = tokenized_prompt[:, 0].long() # [B]        
        # 获取每个样本的有效阶段数
        # 假设 TASK_NUM_STAGES 是一个 List 或 Numpy Array
        # 我们需要把它转成 Tensor 并通过 fancy indexing 取值
        # 注意：你需要确保 TASK_NUM_STAGES 在这里是可以访问的
        if not isinstance(TASK_NUM_STAGES, torch.Tensor):
            task_num_stages_tensor = torch.tensor(TASK_NUM_STAGES, device=subtask_logits.device)
        else:
            task_num_stages_tensor = TASK_NUM_STAGES.to(subtask_logits.device)
        batch_task_num_stages = task_num_stages_tensor[task_ids] # [B]

        MAX_NUM_STAGES = subtask_logits.shape[-1]
        stage_range = torch.arange(MAX_NUM_STAGES, device=subtask_logits.device) # [15]
        valid_mask = stage_range[None, :] < batch_task_num_stages[:, None]

        subtask_logits = torch.where(
            valid_mask, 
            subtask_logits, 
            torch.tensor(float('-inf'), device=subtask_logits.device, dtype=subtask_logits.dtype)
        )
        
        # Mask invalid stages logic for loss calculation
        gt_subtask = tokenized_prompt[:, 1]
        subtask_loss = F.cross_entropy(subtask_logits, gt_subtask.long())
        losses["subtask_loss"] = subtask_loss

        # Calculate accuracy for logging
        with torch.no_grad():
            subtask_acc = (torch.argmax(subtask_logits, dim=-1) == gt_subtask).float().mean()
            losses["subtask_accuracy"] = subtask_acc

        # B. FAST Auxiliary Loss
        fast_loss_value = torch.tensor(0.0, device=prefix_embs.device)
        fast_len = 0
        if self.config.use_fast_auxiliary and fast_tokens is not None:
            fast_len = fast_tokens.shape[1]
            # Extract FAST outputs from the end of prefix sequence
            # prefix_out shape: [B, Seq, Dim]
            fast_outputs = prefix_out[:, -fast_len:, :]

            # Project to vocab
            if fast_outputs.dtype != self.fast_token_proj.weight.dtype:
                fast_outputs = fast_outputs.to(dtype=self.fast_token_proj.weight.dtype)
            fast_logits = self.fast_token_proj(fast_outputs) # [B, T, Vocab]

            # Calculate Cross Entropy with Masking
            # Flatten for CE loss
            vocab_size = fast_logits.shape[-1]
            loss_flat = F.cross_entropy(
                fast_logits.reshape(-1, vocab_size),
                fast_tokens.reshape(-1).long(),
                reduction='none'
            )

            # Apply mask
            mask_flat = fast_token_mask.reshape(-1).float()
            masked_loss = loss_flat * mask_flat

            # Normalize by valid tokens
            num_valid = torch.clamp(mask_flat.sum(), min=1.0)
            avg_fast_loss = masked_loss.sum() / num_valid

            losses["fast_loss"] = avg_fast_loss

            # Compute accuracy
            with torch.no_grad():
                pred_tokens = torch.argmax(fast_logits, dim=-1)
                correct = (pred_tokens == fast_tokens) * fast_token_mask
                losses["fast_accuracy"] = correct.sum() / num_valid

            fast_loss_value = self.config.fast_loss_weight * avg_fast_loss

        # -------------------------------------------------------------------------
        # 4. KV Cache Management (Slice FAST & Transform)
        # -------------------------------------------------------------------------
        # Prepare prefix masks for the Action Expert (Must exclude FAST tokens)
        if fast_len > 0:
            # Slice KV Cache: Remove last fast_len tokens from sequence dimension (dim 2)
            # HF Cache is tuple(k, v), shape [B, H, Seq, D]
            new_past_key_values = []
            for k, v in past_key_values:
                k_sliced = k[:, :, :-fast_len, :]
                v_sliced = v[:, :, :-fast_len, :]
                new_past_key_values.append((k_sliced, v_sliced))
            past_key_values = new_past_key_values

            # Slice masks for Suffix attention
            prefix_pad_masks_for_actions = prefix_pad_masks[:, :-fast_len]
        else:
            prefix_pad_masks_for_actions = prefix_pad_masks

        # Knowledge Insulation (Stop Gradient)
        if self.config.use_knowledge_insulation:
            past_key_values = [
                (k.detach(), v.detach()) for k, v in past_key_values
            ]

        # KV Transform (Mixing across layers)
        if self.kv_transform is not None:
            past_key_values = self.kv_transform(past_key_values)

        # -------------------------------------------------------------------------
        # 5. Multi-Sample Flow Matching Loop
        # -------------------------------------------------------------------------
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]

        # Store all losses to average later: List of [B, H, D] tensors
        sample_losses_list = []

        # Loop for multiple flow samples (reusing the same cached prefix)
        for _ in range(num_flow_samples):
            # A. Sample Noise and Time (if not provided externally)
            # Note: JAX splits RNG key. PyTorch relies on global RNG state usually.
            curr_noise = noise if noise is not None else self.generate_correlated_noise(batch_size, actions.device)
            curr_time = time if time is not None else self.sample_time(batch_size, actions.device)

            # B. Flow Matching Interpolation
            time_expanded = curr_time[:, None, None] # [B, 1, 1]
            x_t = time_expanded * curr_noise + (1 - time_expanded) * actions
            u_t = curr_noise - actions # Target velocity

            # C. Embed Suffix
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                observation, x_t, curr_time
            )

            # D. Construct Attention Masks for Suffix
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks_for_actions.shape[1]

            # Prefix mask expanded for Suffix query
            # [B, Suffix_Len, Prefix_Len]
            prefix_mask_expanded = prefix_pad_masks_for_actions.unsqueeze(1).expand(batch_size, suffix_len, prefix_len)

            # Suffix attending to itself
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

            # Concatenate
            full_att_2d = torch.cat([prefix_mask_expanded, suffix_att_2d], dim=2)
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d)

            # Positions (Offset by prefix length)
            suffix_pos_ids = prefix_len + torch.cumsum(suffix_pad_masks, dim=1) - 1

            if past_key_values is not None and isinstance(past_key_values, list):
                kv_cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(past_key_values):
                    kv_cache.key_cache.append(k)
                    kv_cache.value_cache.append(v)
                past_key_values = kv_cache

            # E. Action Expert Forward (Suffix Only)
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=suffix_pos_ids,
                past_key_values=past_key_values, # The transformed, sliced cache
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond]
            )

            # F. Compute MSE Loss
            # Output: [B, Horizon, Hidden] -> [B, Horizon, Action_Dim]
            suffix_slice = suffix_out[:, -self.config.action_horizon:]
            if suffix_slice.dtype != self.action_out_proj.weight.dtype:
                suffix_slice = suffix_slice.to(dtype=self.action_out_proj.weight.dtype)
            v_pred = self.action_out_proj(suffix_slice)

            # Squared Error per dimension [B, H, D]
            sample_mse = (v_pred - u_t) ** 2
            sample_losses_list.append(sample_mse)

        # -------------------------------------------------------------------------
        # 6. Average and Log Detailed Losses
        # -------------------------------------------------------------------------
        # Stack and Mean over flow samples: [Num_Samples, B, H, D] -> [B, H, D]
        avg_action_loss_tensor = torch.stack(sample_losses_list).mean(dim=0)

        # Mean over Batch and Horizon for final scalar loss
        losses["action_loss"] = avg_action_loss_tensor.mean()

        # --- Detailed Dimension Logging (Matches JAX 1:1) ---
        # Dimensions:
        # 0-2: Base Vel (x,y,z)
        # 3-6: Trunk (4 joints)
        # 7-13: Left Arm (7 joints)
        # 14: Left Gripper
        # 15-21: Right Arm (7 joints)
        # 22: Right Gripper

        # Helper to safely log mean over B and H
        def log_dim_loss(name, tensor_slice):
            losses[name] = tensor_slice.mean()

        # Base Velocity
        log_dim_loss("action_loss_base_vel_x", avg_action_loss_tensor[..., 0])
        log_dim_loss("action_loss_base_vel_y", avg_action_loss_tensor[..., 1])
        log_dim_loss("action_loss_base_vel_z", avg_action_loss_tensor[..., 2])

        # Trunk
        for i in range(4):
            if 3+i < action_dim:
                log_dim_loss(f"action_loss_trunk_{i}", avg_action_loss_tensor[..., 3+i])

        # Left Arm
        for i in range(7):
            if 7+i < action_dim:
                log_dim_loss(f"action_loss_left_arm_{i}", avg_action_loss_tensor[..., 7+i])

        # Left Gripper
        if 14 < action_dim:
            log_dim_loss("action_loss_left_gripper", avg_action_loss_tensor[..., 14])

        # Right Arm
        for i in range(7):
            if 15+i < action_dim:
                log_dim_loss(f"action_loss_right_arm_{i}", avg_action_loss_tensor[..., 15+i])

        # Right Gripper
        if 22 < action_dim:
            log_dim_loss("action_loss_right_gripper", avg_action_loss_tensor[..., 22])

        # -------------------------------------------------------------------------
        # 7. Total Loss Combination
        # -------------------------------------------------------------------------
        losses["total_loss"] = (
            losses["action_loss"] +
            self.config.subtask_loss_weight * losses["subtask_loss"] +
            fast_loss_value
        )

        return losses


    # Override standard forward to use the detailed one
    def forward(self, observation, actions, noise=None, time=None):
        losses = self.forward_detailed(observation, actions, noise, time)
        return losses["total_loss"]

    def load_correlation_matrix(self, norm_stats: Dict):
        """
        Load full correlation matrix from normalization statistics and apply shrinkage.
        Corresponds to JAX `load_correlation_matrix`.
        """
        if not self.use_correlated_noise:
            print("Correlated noise disabled in config, skipping correlation matrix loading")
            return

        if 'actions' not in norm_stats:
            raise ValueError("use_correlated_noise=True but 'actions' key not found in norm_stats.")

        actions_stats = norm_stats['actions']

        # 获取 Cholesky 矩阵 (支持 dict 或 object 属性访问)
        if isinstance(actions_stats, dict):
            chol_matrix = actions_stats.get('action_correlation_cholesky')
        elif hasattr(actions_stats, 'action_correlation_cholesky'):
            chol_matrix = actions_stats.action_correlation_cholesky
        else:
            raise ValueError("Cannot find 'action_correlation_cholesky' in norm_stats['actions']")

        if chol_matrix is None:
            raise ValueError("action_correlation_cholesky is None.")

        # 转换为 PyTorch Tensor
        L = torch.tensor(chol_matrix, dtype=torch.float32, device=self.action_correlation_cholesky.device)

        expected_dim = self.config.action_horizon * self.config.action_dim
        if L.shape[0] != expected_dim or L.shape[1] != expected_dim:
             raise ValueError(f"Correlation matrix shape mismatch. Expected {expected_dim}x{expected_dim}, got {L.shape}")

        # 1. Reconstruct Covariance Matrix: Sigma = L @ L.T
        Sigma = L @ L.T

        # 2. Apply Shrinkage Regularization: Sigma_reg = beta * Sigma + (1-beta) * I
        beta = self.correlation_beta
        print(f"Applying shrinkage regularization with beta={beta:.2f}")

        I = torch.eye(Sigma.shape[0], device=Sigma.device)
        Sigma_reg = beta * Sigma + (1 - beta) * I

        # 3. Compute Cholesky of Regularized Matrix
        try:
            # use linalg.cholesky_ex for better error handling if needed, but standard is fine
            L_reg = torch.linalg.cholesky(Sigma_reg)
        except RuntimeError as e:
            raise RuntimeError(f"Cholesky decomposition failed on regularized covariance: {e}")

        # 更新 Buffer
        self.action_correlation_cholesky.copy_(L_reg)
        self.correlation_loaded = True
        print(f"✓ Loaded correlation matrix with shape {L_reg.shape}")

    def generate_correlated_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate correlated noise matching action covariance structure.
        Corresponds to JAX `generate_correlated_noise`.
        """
        flat_dim = self.config.action_horizon * self.config.action_dim

        # 1. Generate Standard Normal Noise [B, Flat_Dim]
        standard_normal = torch.randn(batch_size, flat_dim, device=device, dtype=self.action_correlation_cholesky.dtype)

        if not self.use_correlated_noise:
            # Fallback to independent noise
            return standard_normal.reshape(batch_size, self.config.action_horizon, self.config.action_dim)

        if not self.correlation_loaded:
             raise RuntimeError(
                "use_correlated_noise=True but correlation matrix is not loaded. "
                "Call load_correlation_matrix() after initialization."
            )

        # 2. Apply Correlation: noise = z @ L.T
        # self.action_correlation_cholesky is L_reg
        correlated_flat = standard_normal @ self.action_correlation_cholesky.t()

        return correlated_flat.reshape(batch_size, self.config.action_horizon, self.config.action_dim)

    def _precompute_correction_matrix(
        self,
        O_indices: torch.Tensor,
        U_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Precompute matrix for correlation-aware inpainting correction.
        Computes Sigma_UO @ Sigma_OO^-1.
        """
        if not self.correlation_loaded:
            raise RuntimeError("Cannot precompute correction matrix: correlation matrix not loaded.")

        # self.action_correlation_cholesky stores the Cholesky factor L (lower triangular)
        L = self.action_correlation_cholesky
        Sigma = L @ L.t() # Reconstruct full covariance [Flat_Dim, Flat_Dim]

        # Extract submatrices using meshgrid-like indexing
        # PyTorch indexing: Sigma[rows[:, None], cols]
        Sigma_OO = Sigma[O_indices[:, None], O_indices] # [|O|, |O|]
        Sigma_UO = Sigma[U_indices[:, None], O_indices] # [|U|, |O|]

        # Regularization for stability
        eps_OO = 1e-6 * torch.max(torch.mean(torch.diag(Sigma_OO)), torch.tensor(1.0, device=Sigma.device))
        Sigma_OO_reg = Sigma_OO + eps_OO * torch.eye(Sigma_OO.shape[0], device=Sigma.device)

        # Solve linear system: X @ Sigma_OO_reg = Sigma_UO.T  =>  Sigma_OO_reg.T @ X.T = Sigma_UO.T
        # We want correction_matrix = Sigma_UO @ Sigma_OO^-1
        # Equivalent to solving A * x = B where A = Sigma_OO, B = Sigma_UO.T, then Transpose result
        # torch.linalg.solve(A, B) solves AX = B

        # Target: C = Sigma_UO @ inv(Sigma_OO)
        # C.T = inv(Sigma_OO).T @ Sigma_UO.T = inv(Sigma_OO) @ Sigma_UO.T
        correction_matrix_T = torch.linalg.solve(Sigma_OO_reg, Sigma_UO.t())
        correction_matrix = correction_matrix_T.t() # [|U|, |O|]

        return {
            'O_indices': O_indices,
            'U_indices': U_indices,
            'correction_matrix': correction_matrix,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        num_steps: int = 10,
        noise: Optional[torch.Tensor] = None,
        initial_actions: Optional[torch.Tensor] = None, # [B, N_steps, N_dims]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference with Correlation-Aware Inpainting.
        """
        device = observation.state.device
        batch_size = observation.state.shape[0]

        # --- 1. Inpainting Setup ---
        fixed_z_O = None
        x0_O = None
        inpainting_data = None

        # Initialize basic params
        action_dim = self.config.action_dim
        action_horizon = self.config.action_horizon
        flat_dim = action_horizon * action_dim

        if initial_actions is not None:
            # Check dimensions
            num_initial_steps = initial_actions.shape[1]
            input_action_dim = initial_actions.shape[2]

            # Pad initial_actions to full model dimensions [B, Horizon, Dim]
            # Padding dims
            if input_action_dim < action_dim:
                padding_dim = torch.zeros(batch_size, num_initial_steps, action_dim - input_action_dim, device=device)
                initial_actions_full = torch.cat([initial_actions, padding_dim], dim=2)
            else:
                initial_actions_full = initial_actions[:, :, :action_dim]

            # Padding steps
            if num_initial_steps < action_horizon:
                padding_steps = torch.zeros(batch_size, action_horizon - num_initial_steps, action_dim, device=device)
                initial_actions_padded = torch.cat([initial_actions_full, padding_steps], dim=1)
            else:
                initial_actions_padded = initial_actions_full[:, :action_horizon]

            # Flatten indices logic
            # O_indices: The flat indices we observe
            O_indices_list = [
                t * action_dim + d
                for t in range(num_initial_steps)
                for d in range(input_action_dim)
            ]
            O_indices = torch.tensor(O_indices_list, dtype=torch.long, device=device)

            # U_indices: The flat indices we need to infer
            O_set = set(O_indices_list)
            U_indices_list = [i for i in range(flat_dim) if i not in O_set]
            U_indices = torch.tensor(U_indices_list, dtype=torch.long, device=device)

            # Generate Correlated Noise
            if noise is None:
                if self.correlation_loaded:
                    noise = self.generate_correlated_noise(batch_size, device)
                else:
                    noise = torch.randn(batch_size, action_horizon, action_dim, device=device)

            # Extract constraints for O (Observed) part
            noise_flat = noise.reshape(batch_size, flat_dim)
            fixed_z_O = noise_flat[:, O_indices] # [B, |O|]
            x0_O = initial_actions_padded.reshape(batch_size, flat_dim)[:, O_indices] # [B, |O|] target actions

            # Precompute Correction Matrix if correlation is available
            if self.correlation_loaded:
                # Cache lookup could be added here similar to JAX,
                # but computing on the fly is usually fast enough for inference.
                inpainting_data = self._precompute_correction_matrix(O_indices, U_indices)
        else:
            # Standard Generation
            if noise is None:
                if self.correlation_loaded:
                    noise = self.generate_correlated_noise(batch_size, device)
                else:
                    noise = torch.randn(batch_size, action_horizon, action_dim, device=device)

        # --- 2. Prefix Encoding & VLM Pass ---
        # Ensure FAST tokens are OFF
        if hasattr(observation, 'fast_tokens') and observation.fast_tokens is not None:
             raise ValueError("FAST tokens must be None during inference.")

        # Preprocess
        images, img_masks, _, _, state = self._preprocess_observation(observation, train=False)
        tokenized_prompt = observation.tokenized_prompt

        # Embed Prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokenized_prompt, state
        )

        # VLM Pass (Get KV Cache)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), kv_cache = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            inputs_embeds=[prefix_embs, None],
            use_cache=True
        )

        # Stage Prediction (Auxiliary Output)
        # Assuming base task token is after images.
        # Calculate image token count to find task start.
        # (Alternatively, pass explicit index if structure is rigid)
        num_img_tokens = sum([img.shape[1] for img in images])
        base_task_token_idx = num_img_tokens
        base_task_output = prefix_out[:, base_task_token_idx, :]
        subtask_logits = self.stage_pred_from_vlm(base_task_output)

        # Mask invalid stages logic (Optional, same as forward)
        # ...

        # Transform KV Cache
        if self.kv_transform is not None:
            kv_cache = self.kv_transform(kv_cache)

        # --- 3. Denoising Loop ---
        dt = -1.0 / num_steps
        x_t = noise
        time = torch.tensor(1.0, device=device)

        # Prepare fixed masks for suffix pass to avoid re-creation
        suffix_len = self.config.action_horizon
        # Suffix mask is all ones (valid)
        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
        # Suffix AR mask: [1] + [0] * (H-1) -> Actually based on embed_suffix:
        # embed_suffix returns att_masks = [1] (for action token block).
        # Wait, embed_suffix logic in previous code:
        # att_masks += [1] + ([0] * (H-1)) -> This means first token is causal?
        # Let's trust embed_suffix return values inside the loop.

        batch_size, prefix_len = prefix_pad_masks.shape

        # Threshold for Inpainting (from Config)
        TIME_THRESHOLD_INPAINT = getattr(self.config, 'time_threshold_inpaint', 0.0)

        step_idx = 0
        while time >= -dt / 2: # Robust float comparison
            # 3.1 Model Step
            suffix_embs, _, suffix_att_masks, adarms_cond = self.embed_suffix(
                observation, x_t, time.expand(batch_size)
            )

            # Construct Attention Masks
            prefix_mask_expanded = prefix_pad_masks.unsqueeze(1).expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_mask_expanded, suffix_att_2d], dim=2)
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d)

            suffix_pos = prefix_len + torch.cumsum(suffix_pad_masks, dim=1) - 1

            # Forward Suffix
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=suffix_pos,
                past_key_values=kv_cache, # Reused cache
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond]
            )

            v_t = self.action_out_proj(suffix_out[:, -action_horizon:])

            # 3.2 Euler Update
            x_t_new = x_t + dt * v_t
            time_new = time + dt

            # 3.3 Inpainting Correction
            if inpainting_data is not None and time_new > TIME_THRESHOLD_INPAINT:
                # Flatten for manipulation
                x_flat = x_t_new.reshape(batch_size, flat_dim)

                # A. Calculate desired state at Observed indices
                # x_desired_O = (1-t) * x0 + t * z
                x_desired_O = (1.0 - time_new) * x0_O + time_new * fixed_z_O

                # B. Calculate Correction at O
                current_O = x_flat[:, inpainting_data['O_indices']]
                delta_O = x_desired_O - current_O

                # C. Force Constraint at O
                # x_flat[:, inpainting_data['O_indices']] = x_desired_O
                # (Use scatter or direct assignment if indices are valid)
                x_flat.scatter_(1, inpainting_data['O_indices'].unsqueeze(0).expand(batch_size, -1), x_desired_O)

                # D. Propagate to Unobserved (U)
                # delta_U = delta_O @ correction_matrix.T
                correction_mat = inpainting_data['correction_matrix'] # [|U|, |O|]
                delta_U = delta_O @ correction_mat.t()

                # E. Stability Check (Clamp large corrections)
                max_correction = torch.max(torch.abs(delta_U))
                if max_correction <= 1.0:
                    # Apply correction to U
                    # x_flat[:, U] += delta_U
                    # Using scatter_add_ or manual indexing
                    current_U = x_flat[:, inpainting_data['U_indices']]
                    x_flat.scatter_(1, inpainting_data['U_indices'].unsqueeze(0).expand(batch_size, -1), current_U + delta_U)
                else:
                    # If unstable, we skip the U correction (but O is already forced)
                    pass

                # Reshape back
                x_t_new = x_flat.reshape(batch_size, action_horizon, action_dim)

            # Update loop vars
            x_t = x_t_new
            time = time_new
            step_idx += 1

        return x_t, subtask_logits


    def embed_suffix(
        self,
        observation, # 修改：这里接收 observation 对象，对应 JAX 的 obs
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed suffix: actions + time (for AdaRMS).

        Args:
            observation: The observation object (ignored in PiBehavior as state is in prefix).
            noisy_actions: [Batch, Horizon, Action_Dim]
            timestep: [Batch]

        Corresponds exactly to JAX `embed_suffix(self, obs, noisy_actions, timestep)`.
        """
        # 1. Embed Timestep (Sine-Cosine)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device
        )
        #time_emb = time_emb.type(dtype=noisy_actions.dtype)
        time_emb = time_emb.to(dtype=self.time_mlp_in.weight.dtype)
        noisy_actions = noisy_actions.to(dtype=self.time_mlp_in.weight.dtype)

        # 2. Time MLP for AdaRMS Conditioning
        def time_mlp_func(t_emb):
            x = self.time_mlp_in(t_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)

        # 3. Embed Actions
        def action_proj_func(actions):
            return self.action_in_proj(actions)

        action_tokens = self._apply_checkpoint(action_proj_func, noisy_actions)

        # 4. Construct Outputs
        embs = action_tokens
        bsize, horizon, _ = embs.shape

        # Pad Mask: All valid
        pad_masks = torch.ones(bsize, horizon, dtype=torch.bool, device=embs.device)

        # Attention Mask: First token causal, rest bidirectional (relative to block)
        # Matches JAX: ar_mask += [True] + ([False] * (self.action_horizon - 1))
        att_masks_list = [1] + [0] * (horizon - 1)
        att_masks = torch.tensor(att_masks_list, dtype=torch.long, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, horizon)

        return embs, pad_masks, att_masks, adarms_cond
