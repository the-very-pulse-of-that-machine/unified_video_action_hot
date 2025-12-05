from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unified_video_action.model.autoregressive.hot_transformer_block import HOTTransformerBlock
from unified_video_action.model.autoregressive.diffusion_loss import DiffLoss
from unified_video_action.model.autoregressive.diffusion_action_loss import DiffActLoss


def mask_by_order(mask_len, order, bsz, seq_len, device):
    masking = torch.zeros(bsz, seq_len).to(device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).to(device),
    ).bool()
    return masking


class MAR(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        diffloss_d=3,
        diffloss_w=1024,
        diffloss_act_d=3,
        diffloss_act_w=1024,
        num_sampling_steps="100",
        diffusion_batch_mul=4,
        grad_checkpointing=False,
        predict_video=True,
        act_diff_training_steps=1000,
        act_diff_testing_steps="100",
        action_model_params={},
        **kwargs
    ):
        super().__init__()

        self.task_name = kwargs["task_name"]
        self.different_history_freq = kwargs["different_history_freq"]
        self.use_history_action = kwargs["use_history_action"]
        self.action_mask_ratio = kwargs["action_mask_ratio"]
        self.use_proprioception = kwargs["use_proprioception"]
        self.predict_wrist_img = kwargs["predict_wrist_img"]
        self.predict_proprioception = kwargs["predict_proprioception"]
        self.n_frames = 4

        # ========= VAE and patchify specifics =========
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.vae_embed_dim = vae_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.label_drop_prob = label_drop_prob

        # ========= Masked MAE =========
        # variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # ========= Projection =========
        # conditional frames
        self.z_proj_cond = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)

        # video frames
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)

        # wrist video frames
        if self.predict_wrist_img:
            self.z_proj_wrist = nn.Linear(
                self.token_embed_dim, encoder_embed_dim, bias=True
            )

        # action
        self.predict_action = action_model_params["predict_action"]
        act_dim = kwargs["shape_meta"]["action"]["shape"][0]

        self.action_proj_cond = nn.Linear(act_dim, encoder_embed_dim, bias=True)
        self.buffer_size_action = 64

        # ========= Fake Latent =========
        self.fake_latent_x = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.fake_action_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        if self.predict_wrist_img:
            self.fake_latent_wrist_x = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        if self.use_history_action:
            self.fake_latent_history_action = nn.Parameter(
                torch.zeros(1, encoder_embed_dim)
            )

        # ========= History Action =========
        if self.use_history_action:
            self.history_action_proj_cond = nn.Linear(
                act_dim, encoder_embed_dim, bias=True
            )

        # ========= Proprioception =========
        if self.use_proprioception:
            self.buffer_size_properception = 64
            if self.different_history_freq:
                self.buffer_size_properception = 64 * 4

            if self.task_name == "umi":
                self.proprioception_proj_cond = nn.Linear(
                    16, encoder_embed_dim, bias=True
                )
            elif "pusht" in self.task_name:
                self.proprioception_proj_cond = nn.Linear(
                    2, encoder_embed_dim, bias=True
                )
            else:
                self.proprioception_proj_cond = nn.Linear(
                    9, encoder_embed_dim, bias=True
                )

            self.proprioception_image_proj_cond = nn.Linear(
                self.token_embed_dim, encoder_embed_dim, bias=True
            )

        # ========= Language Embedding =========
        self.language_emb_model = kwargs["language_emb_model"]
        self.language_emb_model_type = 1

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
                self.text_proj_cond = nn.Linear(
                    512, encoder_embed_dim, bias=True
                )  # clip text embedding is 512
                self.buffer_size_text = 64
                self.text_pos_embed = nn.Parameter(
                    torch.zeros(1, self.buffer_size_text, encoder_embed_dim)
                )

        # ========= Projection =========
        if self.predict_wrist_img:
            proj_cond_x_dim_num = 4
            if self.use_proprioception:
                proj_cond_x_dim_num += 2
            if self.use_history_action:
                proj_cond_x_dim_num += 1
        else:
            proj_cond_x_dim_num = 3
            if self.use_proprioception:
                if (
                    self.task_name == "umi"
                    or "block_push" in self.task_name
                    or "pusht" in self.task_name
                ):
                    proj_cond_x_dim_num += 1
                else:
                    proj_cond_x_dim_num += 2
            if self.use_history_action:
                proj_cond_x_dim_num += 1

        self.proj_cond_x_layer = nn.Linear(
            proj_cond_x_dim_num * encoder_embed_dim, encoder_embed_dim, bias=True
        )

        # ========= Temporal and Spatial Position Embedding =========
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, encoder_embed_dim)
        )  # Temporal position embedding, 4 frames
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, encoder_embed_dim)
        )  # Spatial position embedding

        # ========= Normalization =========
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # ========= Encoder Blocks =========
        self.encoder_blocks = nn.ModuleList(
            [
                HOTTransformerBlock(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                    use_hot=True
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # ========= Decoder =========
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # ========= Decoder Position Embedding =========
        self.decoder_temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, decoder_embed_dim)
        )
        self.decoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        # ========= Decoder Text Position Embedding =========
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                self.decoder_text_pos_embed = nn.Parameter(
                    torch.zeros(1, self.buffer_size_text, decoder_embed_dim)
                )

        # ========= Decoder Blocks =========
        self.decoder_blocks = nn.ModuleList(
            [
                HOTTransformerBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                    use_hot=True
                )
                for _ in range(decoder_depth)
            ]
        )

        # ========= Decoder Norm =========
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # ========= Diffusion Temporal and Spatial Embedding =========
        self.diffusion_temporal_embed = nn.Parameter(
            torch.zeros(1, self.n_frames, decoder_embed_dim)
        )
        self.diffusion_spatial_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        # ========= Initialize Weights =========
        self.initialize_weights()

        # ========= Video Diffusion Loss =========
        self.predict_video = predict_video
        if self.predict_video:
            # ========= Video Diffusion Loss =========
            self.diffloss = DiffLoss(
                target_channels=self.token_embed_dim,
                z_channels=decoder_embed_dim,
                width=diffloss_w,
                depth=diffloss_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                n_frames=self.n_frames,
                language_emb_model=self.language_emb_model,
                language_emb_model_type=self.language_emb_model_type,
            )

            # ========= Wrist Video Diffusion Loss =========
            if self.predict_wrist_img:
                self.diffloss_wrist = DiffLoss(
                    target_channels=self.token_embed_dim,
                    z_channels=decoder_embed_dim,
                    width=diffloss_w,
                    depth=diffloss_d,
                    num_sampling_steps=num_sampling_steps,
                    grad_checkpointing=grad_checkpointing,
                    n_frames=self.n_frames,
                    language_emb_model=self.language_emb_model,
                    language_emb_model_type=self.language_emb_model_type,
                )

        # ========= Action Diffusion Loss =========
        if self.predict_action:
            self.diffactloss = DiffActLoss(
                target_channels=act_dim,
                z_channels=decoder_embed_dim,
                width=diffloss_act_w,
                depth=diffloss_act_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                n_frames=self.n_frames,
                act_model_type=action_model_params["act_model_type"],
                act_diff_training_steps=act_diff_training_steps,
                act_diff_testing_steps=act_diff_testing_steps,
                language_emb_model=self.language_emb_model,
                language_emb_model_type=self.language_emb_model_type,
            )

        
        # ========= Proprioception Diffusion Loss =========
        if self.predict_proprioception:
            if self.task_name == "umi":
                self.diffproploss = DiffActLoss(
                    target_channels=6,
                    z_channels=decoder_embed_dim,
                    width=diffloss_act_w,
                    depth=diffloss_act_d,
                    num_sampling_steps=num_sampling_steps,
                    grad_checkpointing=grad_checkpointing,
                    n_frames=self.n_frames,
                    act_model_type=action_model_params["act_model_type"],
                    act_diff_training_steps=act_diff_training_steps,
                    act_diff_testing_steps=act_diff_testing_steps,
                    language_emb_model=self.language_emb_model,
                    language_emb_model_type=self.language_emb_model_type,
                )
            elif self.task_name == 'toolhang':
                self.diffproploss = DiffActLoss(
                        target_channels=9,
                        z_channels=decoder_embed_dim,
                        width=diffloss_act_w,
                        depth=diffloss_act_d,
                        num_sampling_steps=num_sampling_steps,
                        grad_checkpointing=grad_checkpointing,
                        n_frames=self.n_frames,
                        act_model_type=action_model_params["act_model_type"],
                        act_diff_training_steps=act_diff_training_steps,
                        act_diff_testing_steps=act_diff_testing_steps,
                        language_emb_model=self.language_emb_model,
                        language_emb_model_type=self.language_emb_model_type,
                    )
            else:
                raise NotImplementedError
            

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.fake_latent_x, std=0.02)
        torch.nn.init.normal_(self.fake_action_latent, std=0.02)

        if self.predict_wrist_img:
            torch.nn.init.normal_(self.fake_latent_wrist_x, std=0.02)

        if self.use_history_action:
            torch.nn.init.normal_(self.fake_latent_history_action, std=0.02)

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                torch.nn.init.normal_(self.fake_latent, std=0.02)

        torch.nn.init.normal_(self.temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.spatial_pos_embed, std=0.02)

        torch.nn.init.normal_(self.decoder_temporal_pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_spatial_pos_embed, std=0.02)

        torch.nn.init.normal_(self.diffusion_temporal_embed, std=0.02)
        torch.nn.init.normal_(self.diffusion_spatial_embed, std=0.02)

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                torch.nn.init.normal_(self.text_pos_embed, std=0.02)
                torch.nn.init.normal_(self.decoder_text_pos_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.device).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, t, seq_len, embed_dim = x.shape

        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, t, seq_len, device=x.device)
        # Create the spatial mask for one frame (t=1)
        spatial_mask = torch.zeros(bsz, seq_len, device=x.device)
        spatial_mask = torch.scatter(
            spatial_mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        # Expand the spatial mask to all frames in the video
        mask = spatial_mask.unsqueeze(1).expand(-1, t, -1)

        return mask
    

    def forward_mae_encoder(
        self,
        x,                      # 输入视频特征: [B, T, S, C]
        mask,                   # 掩码: [B, T, S]
        cond,                   # 条件图像 patch: [B, T, S, C]
        text_latents=None,      # 文本 latent
        history_nactions=None,  # 历史动作（可选）
        nactions=None,          # 当前动作（可选）
        task_mode=None,         # 任务模式: policy_model / inverse_model / video_model …
        proprioception_input={},# 本体感知输入
    ):
        # x shape: [B, T, S, C]
        B, T, S, _ = x.size()

        # 掩码展开: [B, T, S] → [B, T*S]
        mask = rearrange(mask, "b t s -> b (t s)")

        # ---------------------------------------------------------------------
        # 🟦 1. 根据任务模式处理 x（视频 latent）与 cond（条件 latent）
        # ---------------------------------------------------------------------
        if task_mode == "policy_model":
            # cond: 投影 + 展平为 [B, T*S, C]
            cond = self.z_proj_cond(cond)
            cond = rearrange(cond, "b t s c -> b (t s) c")

            # 策略模型不使用原始 x，用 fake latent 替代
            x = self.fake_latent_x.unsqueeze(1).expand(B, cond.size(1), -1)

            # 如果预测腕部视频，wrist_x 也使用 fake latent
            if self.predict_wrist_img:
                wrist_x = self.fake_latent_wrist_x.unsqueeze(1).expand(
                    B, cond.size(1), -1
                )

        elif task_mode == "inverse_model":
            # inverse model 使用输入视频 x
            x = self.z_proj(x)
            x = rearrange(x, "b t s c -> b (t s) c")

            # wrist 图像也投影成 latent
            if self.predict_wrist_img:
                wrist_x = self.z_proj_wrist(proprioception_input["pred_second_image_z"])
                wrist_x = rearrange(wrist_x, "b t s c -> b (t s) c")

            # cond 反过来使用 fake latent
            cond = self.fake_latent_x.unsqueeze(1).expand(B, x.size(1), -1)

        else:
            # video_model / dynamic_model / full_dynamic_model 等
            cond = self.z_proj_cond(cond)
            cond = rearrange(cond, "b t s c -> b (t s) c")

            # x 也投影
            x = self.z_proj(x)
            x = rearrange(x, "b t s c -> b (t s) c")

            # mask 位置替换为 fake latent（MAE 的 mask 机制）
            fake_latent_expanded = self.fake_latent_x.unsqueeze(1).expand(
                B, x.size(1), -1
            )
            x[mask == 1] = fake_latent_expanded[mask == 1].to(x.dtype)

            # wrist 也执行同样的 masking
            if self.predict_wrist_img:
                wrist_x = self.z_proj_wrist(proprioception_input["pred_second_image_z"])
                wrist_x = rearrange(wrist_x, "b t s c -> b (t s) c")
                fake_wrist_latent_expanded = self.fake_latent_wrist_x.unsqueeze(
                    1
                ).expand(B, wrist_x.size(1), -1)
                wrist_x[mask == 1] = fake_wrist_latent_expanded[mask == 1].to(
                    wrist_x.dtype
                )

        embed_dim = cond.size(2)

        # ---------------------------------------------------------------------
        # 🟦 2. 历史动作编码（可选）
        # ---------------------------------------------------------------------
        if self.use_history_action:
            if history_nactions is None:
                # 无历史动作 → 用 fake latent
                history_action_latents = self.fake_latent_history_action.unsqueeze(
                    0
                ).repeat(B, T * self.n_frames, 1)
            else:
                # 有历史动作 → 投影
                history_action_latents = self.history_action_proj_cond(history_nactions)

                # 训练时随机 mask 动作
                if self.training:
                    history_action_mask = (
                        torch.rand(B, T * self.n_frames) > self.action_mask_ratio
                    ).int()
                    history_action_latents[history_action_mask == 1] = (
                        self.fake_latent_history_action.to(history_action_latents.dtype)
                    )

            # 扩展到 buffer（重复多次）
            history_action_latents_expand = history_action_latents.repeat_interleave(
                self.buffer_size_action, dim=1
            )

        # ---------------------------------------------------------------------
        # 🟦 3. 本体感知（Proprioception）部分
        # ---------------------------------------------------------------------
        if self.use_proprioception:
            if self.task_name == "umi":
                # UMI robot 特有的 state 构成
                proprioception_state_cond = torch.cat(
                    [
                        proprioception_input["robot0_eef_pos"],
                        proprioception_input["robot0_eef_rot_axis_angle"],
                        proprioception_input["robot0_gripper_width"],
                        proprioception_input["robot0_eef_rot_axis_angle_wrt_start"],
                    ],
                    dim=-1,
                )
                proprioception_state_cond = self.proprioception_proj_cond(
                    proprioception_state_cond.float()
                )
                proprioception_state_cond_expand = (
                    proprioception_state_cond.repeat_interleave(
                        self.buffer_size_properception, dim=1
                    )
                )
            else:
                # 视觉本体信息 second_image_z
                proprioception_image_cond = self.proprioception_image_proj_cond(
                    proprioception_input["second_image_z"]
                )
                proprioception_image_cond = rearrange(
                    proprioception_image_cond, "b t s c -> b (t s) c"
                )

                # eef pos + quat + gripper
                proprioception_state_cond = torch.cat(
                    [
                        proprioception_input["robot0_eef_pos"],
                        proprioception_input["robot0_eef_quat"],
                        proprioception_input["robot0_gripper_qpos"],
                    ],
                    dim=-1,
                )
                proprioception_state_cond = self.proprioception_proj_cond(
                    proprioception_state_cond
                )
                proprioception_state_cond_expand = (
                    proprioception_state_cond.repeat_interleave(
                        self.buffer_size_properception, dim=1
                    )
                )

        # ---------------------------------------------------------------------
        # 🟦 4. 动作 latent
        # ---------------------------------------------------------------------
        if task_mode == "dynamic_model":
            action_latents = self.action_proj_cond(nactions)
        else:
            action_latents = self.fake_action_latent.unsqueeze(0).repeat(B, 16, 1)

        action_latents_expand = action_latents.repeat_interleave(
            self.buffer_size_action, dim=1
        )

        # ---------------------------------------------------------------------
        # 🟦 5. 拼接所有模态（视频、cond、动作、本体、历史动作…）
        # ---------------------------------------------------------------------
        if self.predict_wrist_img:
            parts = [x, wrist_x, cond]
            if self.use_history_action:
                parts.append(history_action_latents_expand)
            parts.append(action_latents_expand)
            if self.use_proprioception:
                parts.extend(
                    [proprioception_image_cond, proprioception_state_cond_expand]
                )
            x = torch.cat(parts, dim=-1)
        else:
            parts = [x, cond]
            if self.use_history_action:
                parts.append(history_action_latents_expand)
            parts.append(action_latents_expand)

            if self.use_proprioception:
                if self.task_name == "umi":
                    parts.append(proprioception_state_cond_expand)
                else:
                    parts.extend(
                        [proprioception_image_cond, proprioception_state_cond_expand]
                    )
            x = torch.cat(parts, dim=-1)

        # ---------------------------------------------------------------------
        # 🟦 6. 模态融合 Projection
        # ---------------------------------------------------------------------
        x = self.proj_cond_x_layer(x)

        # ---------------------------------------------------------------------
        # 🟦 7. 时空位置编码
        # ---------------------------------------------------------------------
        temporal_pos_embed_expanded = self.temporal_pos_embed.unsqueeze(2).expand(
            -1, -1, S, -1
        )
        spatial_pos_embed_expanded = self.spatial_pos_embed.unsqueeze(1).expand(
            -1, T, -1, -1
        )
        combined_pos_embed = (
            temporal_pos_embed_expanded + spatial_pos_embed_expanded
        ).reshape(-1, T * S, embed_dim)

        x = x + combined_pos_embed

        # ---------------------------------------------------------------------
        # 🟦 8. 文本 embedding 插入（用于条件生成）
        # ---------------------------------------------------------------------
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                text_latents = text_latents.unsqueeze(1).repeat(
                    1, self.buffer_size_text, 1
                )

                # classifier free guidance (CFG)
                if self.training:
                    drop_latent_mask = torch.rand(B) < self.label_drop_prob
                    drop_latent_mask = (
                        drop_latent_mask.unsqueeze(-1).to(self.device).to(x.dtype)
                    )
                    drop_latent_mask = drop_latent_mask.unsqueeze(1).repeat(
                        1, self.buffer_size_text, 1
                    )
                    text_latents = (
                        drop_latent_mask
                        * self.fake_latent.unsqueeze(1).repeat(
                            1, self.buffer_size_text, 1
                        )
                        + (1 - drop_latent_mask) * text_latents
                    )

                text_latents = text_latents + self.text_pos_embed
                x = torch.cat([text_latents, x], dim=1)

        # ---------------------------------------------------------------------
        # 🟦 9. LayerNorm
        # ---------------------------------------------------------------------
        x = self.z_proj_ln(x)

        # ---------------------------------------------------------------------
        # 🟦 10. Transformer Encoder Blocks（可 checkpoint）
        # ---------------------------------------------------------------------
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)

        # 最终编码后的序列
        x = self.encoder_norm(x)

        return x


    def forward_mae_decoder(self, x, mask):
        # mask 形状为 [B, T, S]
        B, T, S = mask.size()

        # 展平 mask: [B, T, S] -> [B, T*S]
        mask = rearrange(mask, "b t s -> b (t s)")

        # 将 encoder 输出的特征 x 做 decoder 输入投影（降维或换通道）
        # x shape remains: [B, T*S (+ text), C]
        x = self.decoder_embed(x)
        _, _, embed_dim = x.shape

        # ---------------------------------------------------------------------
        # 🟦 1. MAE Decoder 的时空位置编码（与 Encoder 独立）
        # ---------------------------------------------------------------------

        # decoder_temporal_pos_embed：形状 [1, T, C]
        # 扩展到 [1, T, S, C]
        decoder_temporal_pos_embed_expanded = self.decoder_temporal_pos_embed.unsqueeze(
            2
        ).expand(
            -1, -1, S, -1
        )

        # decoder_spatial_pos_embed：形状 [1, S, C]
        # 扩展到 [1, T, S, C]
        decoder_spatial_pos_embed_expanded = self.decoder_spatial_pos_embed.unsqueeze(
            1
        ).expand(
            -1, T, -1, -1
        )

        # 两者相加，得到每个 patch 对应的时空位置编码
        # reshape → [1, T*S, C]
        decoder_combined_pos_embed = (
            decoder_temporal_pos_embed_expanded + decoder_spatial_pos_embed_expanded
        ).reshape(1, T * S, embed_dim)

        # ---------------------------------------------------------------------
        # 🟦 2. 若有文本 embedding，则在 decoder 也要处理位置编码
        # ---------------------------------------------------------------------

        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                # text embedding 的 decoder 位置编码 concat 在最前
                combined_pos_embed = torch.cat(
                    [self.decoder_text_pos_embed, decoder_combined_pos_embed],
                    dim=1,
                )
            else:
                combined_pos_embed = decoder_combined_pos_embed
        else:
            combined_pos_embed = decoder_combined_pos_embed

        # 加上 decoder 的位置编码
        x = x + combined_pos_embed

        # ---------------------------------------------------------------------
        # 🟦 3. Transformer Decoder Blocks
        # ---------------------------------------------------------------------
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)

        # Decoder 层标准化
        x = self.decoder_norm(x)

        # ---------------------------------------------------------------------
        # 🟦 4. 剔除文本部分（若 decoder 前面拼接了文本 token）
        # ---------------------------------------------------------------------
        if self.language_emb_model == "clip":
            if self.language_emb_model_type == 1:
                # x 前 buffer_size_text 个 token 是文本特征，不参与重建
                # 保留后续的视频 token
                x = x[:, self.buffer_size_text:]

        # ---------------------------------------------------------------------
        # 🟦 5. diffusion（扩散模型）用的额外位置编码
        # ---------------------------------------------------------------------
        # 扩散模型用自己的时空编码，与 MAE 位置编码分离

        diffusion_temporal_pos_embed_expanded = self.diffusion_temporal_embed.unsqueeze(
            2
        ).expand(
            -1, -1, S, -1
        )

        diffusion_spatial_pos_embed_expanded = self.diffusion_spatial_embed.unsqueeze(
            1
        ).expand(
            -1, T, -1, -1
        )

        # 形状: [1, T*S, C]
        diffusion_combined_pos_embed = (
            diffusion_temporal_pos_embed_expanded + diffusion_spatial_pos_embed_expanded
        ).reshape(1, T * S, embed_dim)

        # diffusion 位置编码相加。注意这里不是替换，而是二次加成。
        x = x + diffusion_combined_pos_embed

        # decoder 输出序列
        return x


    def forward_loss(
        self,
        z,                  # decoder 输出的 latent 序列（预测结果）
        target,             # ground truth 重建目标（视频 patch latent）
        mask,               # mask: [B, T, S]，表示被 mask 的 patch
        nactions=None,      # ground truth 动作（用于 policy / inverse / full）
        task_mode=None,     # 当前任务模式：video / policy / inverse / full
        gt_wrist_latents=None,   # wrist camera 的 ground truth latent
        gt_properception=None,   # 本体感知 ground truth (eef pos/quat/gripper)
        text_latents=None,       # 文本 latent，用于条件扩散损失
    ):
        # ----------------------------------------------------------------------
        # 🟦 1. Video Model / Dynamic Model —— 只训练 video（预测未来帧）
        # ----------------------------------------------------------------------
        if task_mode == "video_model" or task_mode == "dynamic_model":

            if self.predict_wrist_img:
                # 主视角视频的 diffusion 重建损失
                video_loss = self.diffloss(
                    z=z,
                    target=target,
                    mask=mask,
                    text_latents=text_latents,
                )
                # 腕部相机视频的 diffusion 重建损失（额外分支）
                video_loss_wrist = self.diffloss_wrist(
                    z=z,
                    target=gt_wrist_latents,
                    mask=mask,
                    text_latents=text_latents,
                )
                # 两者相加：最终视频损失
                video_loss = video_loss + video_loss_wrist
            else:
                # 没有 wrist camera，只有主视频损失
                video_loss = self.diffloss(
                    z=z,
                    target=target,
                    mask=mask,
                    text_latents=text_latents,
                )

            # video model 不训练动作
            act_loss = torch.tensor(0.0).to(self.device)

            # 总损失 = video loss
            loss = video_loss

        # ----------------------------------------------------------------------
        # 🟦 2. Policy Model / Inverse Model —— 只训练动作，不训练视频
        # ----------------------------------------------------------------------
        elif task_mode == "policy_model" or task_mode == "inverse_model":

            # diffusion-based 动作预测损失
            act_loss = self.diffactloss(
                z=z,
                target=nactions,
                task_mode=task_mode,
                text_latents=text_latents,
            )

            # 不做视频重建
            video_loss = torch.tensor(0.0).to(self.device)

            # 总损失只包含动作损失
            loss = act_loss

        # ----------------------------------------------------------------------
        # 🟦 3. Full Dynamic Model —— 同时预测视频（未来帧）+ 动作
        # ----------------------------------------------------------------------
        elif task_mode == "full_dynamic_model":

            # 视频部分
            if self.predict_wrist_img:
                video_loss = self.diffloss(
                    z=z,
                    target=target,
                    mask=mask,
                    text_latents=text_latents,
                )
                video_loss_wrist = self.diffloss_wrist(
                    z=z,
                    target=gt_wrist_latents,
                    mask=mask,
                    text_latents=text_latents,
                )
                video_loss = video_loss + video_loss_wrist
            else:
                video_loss = self.diffloss(
                    z=z,
                    target=target,
                    mask=mask,
                    text_latents=text_latents,
                )

            # 动作部分
            act_loss = self.diffactloss(
                z=z,
                target=nactions,
                task_mode=task_mode,
                text_latents=text_latents,
            )

            # full model = video_loss + action_loss
            loss = video_loss + act_loss

        # ----------------------------------------------------------------------
        # 🟦 4. 若需要预测本体感知 proprioception，则加入额外损失
        # ----------------------------------------------------------------------
        if self.predict_proprioception:
            properception_loss = self.diffproploss(
                z=z,
                target=gt_properception,
                text_latents=text_latents,
            )
            loss = loss + properception_loss

        # 返回：总损失 + video_loss + act_loss
        return loss, video_loss, act_loss

    def forward(
        self,
        imgs,                       # 输入视频 [B, T, C, H, W]
        cond,                       # 条件视频（通常是观测）[B, T, C, H, W]
        history_nactions=None,      # 历史动作
        nactions=None,              # 当前动作（用于 actor 或 inverse）
        text_latents=None,          # 文本 latent
        task_mode=None,             # 当前任务模式
        proprioception_input={},    # 本体感知信息（eef pos/quat/gripper）
    ):
        # 当前 batch 的 device
        self.device = cond.device

        # imgs: [B, T, C, H, W]
        B, T, C, H, W = imgs.size()

        # ----------------------------------------------------------------------
        # 🟦 1. Patchify 视频输入 imgs
        # ----------------------------------------------------------------------

        # 展平 batch 和 time: [B*T, C, H, W]
        imgs = rearrange(imgs, "b t c h w -> (b t) c h w")

        # patchify: -> [B*T, S, C']  (S=patch数)
        x = self.patchify(imgs)

        # 再 reshape 回三维 time: [B, T, S, C']
        x = rearrange(x, "(b t) seq_len c -> b t seq_len c", b=B)

        # ----------------------------------------------------------------------
        # 🟦 2. Patchify 条件 cond（与上面类似）
        # ----------------------------------------------------------------------
        cond = rearrange(cond, "b t c h w -> (b t) c h w")
        cond = self.patchify(cond)
        cond = rearrange(cond, "(b t) seq_len c -> b t seq_len c", b=B)

        # ----------------------------------------------------------------------
        # 🟦 3. 处理 proprioception（若使用第二摄像头）
        # ----------------------------------------------------------------------
        if self.use_proprioception:
            if "second_image_z" in proprioception_input:

                # patchify 第二相机图像
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"], "b t c h w -> (b t) c h w"
                )
                proprioception_input["second_image_z"] = self.patchify(
                    proprioception_input["second_image_z"]
                )
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        # ----------------------------------------------------------------------
        # 🟦 4. 若需要预测 Wrist Camera，则 patchify wrist camera 预测图像
        # ----------------------------------------------------------------------
        if self.predict_wrist_img:
            if "pred_second_image_z" in proprioception_input:
                proprioception_input["pred_second_image_z"] = rearrange(
                    proprioception_input["pred_second_image_z"],
                    "b t c h w -> (b t) c h w",
                )
                proprioception_input["pred_second_image_z"] = self.patchify(
                    proprioception_input["pred_second_image_z"]
                )
                proprioception_input["pred_second_image_z"] = rearrange(
                    proprioception_input["pred_second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        # ----------------------------------------------------------------------
        # 🟦 5. 文本 latent 处理（如使用 text_proj_cond）
        # ----------------------------------------------------------------------
        if text_latents is not None and hasattr(self, "text_proj_cond"):
            if self.language_emb_model_type == 1:
                text_latents = self.text_proj_cond(text_latents)

        # ground truth latent（视频 patch latent）
        gt_latents = x.clone().detach()

        # wrist camera GT latent
        if self.predict_wrist_img:
            if "pred_second_image_z" in proprioception_input:
                gt_wrist_latents = proprioception_input["pred_second_image_z"].clone().detach()
                gt_wrist_latents = rearrange(gt_wrist_latents, "b t s c -> b (t s) c")

        # ----------------------------------------------------------------------
        # 🟦 6. 随机 Mask（MAE）生成 mask
        # ----------------------------------------------------------------------
        # orders shape example: [B, T, S] 的随机遍历顺序
        orders = self.sample_orders(bsz=B)

        # mask: [B, T, S]（1 = masked，0 = keep）
        mask = self.random_masking(x, orders)

        # ----------------------------------------------------------------------
        # 🟦 7. MAE Encoder
        # ----------------------------------------------------------------------
        x = self.forward_mae_encoder(
            x,
            mask,
            cond,
            text_latents,
            history_nactions,
            nactions,
            task_mode=task_mode,
            proprioception_input=proprioception_input,
        )

        # ----------------------------------------------------------------------
        # 🟦 8. MAE Decoder
        # ----------------------------------------------------------------------
        z = self.forward_mae_decoder(x, mask)

        # ----------------------------------------------------------------------
        # 🟦 9. reshape mask / gt (flatten: [B, T, S] → [B, T*S])
        # ----------------------------------------------------------------------
        mask = rearrange(mask, "b t s -> b (t s)")
        gt_latents = rearrange(gt_latents, "b t s c -> b (t s) c")

        # ----------------------------------------------------------------------
        # 🟦 10. 若预测本体感知，则根据任务类型取 GT proprioception
        # ----------------------------------------------------------------------
        if self.predict_proprioception:
            if self.task_name == "umi":
                # UMI 使用 axis-angle rot
                gt_properception = proprioception_input[
                    "robot0_eef_rot_axis_angle_wrt_start_pred"
                ]

            elif self.task_name == "toolhang":
                # toolhang 拼接 eef pos + quat + gripper 状态
                gt_properception = torch.cat([
                    proprioception_input['robot0_eef_pos_pred'],
                    proprioception_input['robot0_eef_quat_pred'],
                    proprioception_input['robot0_gripper_qpos_pred'],
                ], dim=-1)

            else:
                raise NotImplementedError

            # ------------------------------------------------------------------
            # 🟦 11. 计算 Loss：视频 + 动作 + wrist + proprioception
            # ------------------------------------------------------------------
            if self.predict_wrist_img:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_wrist_latents=gt_wrist_latents,
                    gt_properception=gt_properception,
                    text_latents=text_latents,
                )
            else:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_properception=gt_properception,
                    text_latents=text_latents,
                )

        # ----------------------------------------------------------------------
        # 🟦 12. 不预测 proprioception（仅视频 + wrist + 动作）
        # ----------------------------------------------------------------------
        else:
            if self.predict_wrist_img:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    gt_wrist_latents=gt_wrist_latents,
                    text_latents=text_latents,
                )
            else:
                loss, video_loss, act_loss = self.forward_loss(
                    z=z,
                    target=gt_latents,
                    mask=mask,
                    nactions=nactions,
                    task_mode=task_mode,
                    text_latents=text_latents,
                )

        # 返回各项 loss
        return loss, video_loss, act_loss


    def sample_tokens(
        self,
        bsz,
        cond,
        text_latents=None,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        temperature=1.0,
        progress=False,
        history_nactions=None,
        nactions=None,
        proprioception_input={},
        task_mode=None,
        vae_model=None,
        x=None,
    ):
        self.device = cond.device
        B, T, C, H, W = cond.size()
        cond = rearrange(cond, "b t c h w -> (b t) c h w")
        cond = self.patchify(cond)
        cond = rearrange(
            cond, "(b t) seq_len c -> b t seq_len c", b=B
        )

        # ========= Proprioception =========
        if self.use_proprioception:
            if "second_image_z" in proprioception_input:
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"], "b t c h w -> (b t) c h w"
                )
                proprioception_input["second_image_z"] = self.patchify(
                    proprioception_input["second_image_z"]
                )
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        if text_latents is not None and hasattr(self, "text_proj_cond"):
            if self.language_emb_model_type == 1:
                text_latents = self.text_proj_cond(text_latents)

        # ========= Mask =========
        if task_mode == "inverse_model":
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.patchify(x)
            tokens = rearrange(
                x, "(b t) seq_len c -> b t seq_len c", b=B
            )
            mask = torch.zeros(bsz, self.n_frames, self.seq_len).to(self.device)
        else:
            # init and sample generation orders
            tokens = torch.zeros(
                bsz, self.n_frames, self.seq_len, self.token_embed_dim
            ).to(self.device)
            mask = torch.ones(bsz, self.n_frames, self.seq_len).to(self.device)
            if self.predict_wrist_img:
                proprioception_input["pred_second_image_z"] = torch.zeros(
                    bsz, self.n_frames, self.seq_len, self.token_embed_dim
                ).to(self.device)

        # ========= Sample Orders =========
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # ========= Predict Video =========
        if self.predict_video:
            for step in indices:
                cur_tokens = tokens.clone()

                if self.predict_wrist_img:
                    cur_wrist_tokens = proprioception_input[
                        "pred_second_image_z"
                    ].clone()

                x = self.forward_mae_encoder(
                    tokens,
                    mask,
                    cond,
                    text_latents,
                    history_nactions=history_nactions,
                    nactions=nactions,
                    task_mode=task_mode,
                    proprioception_input=proprioception_input,
                )
                z = self.forward_mae_decoder(x, mask)

                if self.predict_action:
                    act_cfg = 1.0
                    sampled_token_latent_act = self.diffactloss.sample(
                        z, temperature, cfg=act_cfg, text_latents=text_latents
                    )
                else:
                    sampled_token_latent_act = None

                # ========= Predict action and return if task_mode is inverse_model or policy_model=========
                if task_mode == "inverse_model" or task_mode == "policy_model":
                    if sampled_token_latent_act is None:
                        # If predict_action is False, create a dummy action tensor
                        # with the expected shape for the action dimension
                        action_dim = self.diffactloss.target_channels if hasattr(self, 'diffactloss') else 7  # fallback to typical action dim
                        sampled_token_latent_act = torch.zeros(
                            bsz, self.n_frames, action_dim
                        ).to(self.device)
                    return None, sampled_token_latent_act

                # ========= Mask Ratio =========
                # mask ratio for the next round, following MaskGIT and MAGE.
                mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
                mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(
                    self.device
                )

                # take the first frame mask
                mask_ = mask[:, 0]

                # masks out at least one for the next iteration
                mask_len = torch.maximum(
                    torch.Tensor([1]).to(self.device),
                    torch.minimum(
                        torch.sum(mask_, dim=-1, keepdims=True) - 1, mask_len
                    ),
                )

                # get masking for next iteration and locations to be predicted in this iteration
                mask_next = mask_by_order(
                    mask_len[0], orders, bsz, self.seq_len, self.device
                )

                ## expand mask_next to all frames
                mask_next = mask_next.unsqueeze(1).expand(-1, T, -1)
                mask_next = rearrange(mask_next, "b t s -> b (t s)")
                mask = rearrange(mask, "b t s -> b (t s)")

                if step >= num_iter - 1:
                    mask_to_pred = mask[:bsz].bool()
                else:
                    mask_to_pred = torch.logical_xor(
                        mask[:bsz].bool(), mask_next.bool()
                    )
                mask = mask_next
                mask = rearrange(mask, "b (t s) -> b t s", t=self.n_frames)

                if not cfg == 1.0:
                    mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

                # sample token latents for this step
                z = z[mask_to_pred.nonzero(as_tuple=True)]
                # cfg schedule follow Muse
                if cfg_schedule == "linear":
                    cfg_iter = (
                        1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
                    )
                elif cfg_schedule == "constant":
                    cfg_iter = cfg
                else:
                    raise NotImplementedError

                sampled_token_latent = self.diffloss.sample(
                    z, temperature, cfg_iter, text_latents=text_latents
                )

                if not cfg == 1.0:
                    sampled_token_latent, _ = sampled_token_latent.chunk(
                        2, dim=0
                    )  # Remove null class samples
                    mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

                cur_tokens = rearrange(cur_tokens, "b t s c -> b (t s) c")
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
                cur_tokens = rearrange(
                    cur_tokens, "b (t s) c -> b t s c", t=self.n_frames
                )
                tokens = cur_tokens.clone()

                # ========= Predict Wrist Image =========
                if self.predict_wrist_img:
                    sampled_wrist_token_latent = self.diffloss_wrist.sample(
                        z, temperature, cfg_iter, text_latents=text_latents
                    )

                    if not cfg == 1.0:
                        sampled_wrist_token_latent, _ = (
                            sampled_wrist_token_latent.chunk(2, dim=0)
                        )  # Remove null class samples

                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b t s c -> b (t s) c"
                    )
                    cur_wrist_tokens[mask_to_pred.nonzero(as_tuple=True)] = (
                        sampled_wrist_token_latent
                    )
                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b (t s) c -> b t s c", t=self.n_frames
                    )
                    proprioception_input["pred_second_image_z"] = (
                        cur_wrist_tokens.clone()
                    )

            # ========= Unpatchify =========
            tokens = rearrange(tokens, "b t s c -> (b t) s c")
            tokens = self.unpatchify(tokens)
            # tokens = rearrange(tokens, '(b t) c h w -> b t c h w', b=B)

            if self.predict_wrist_img:
                wrist_tokens = rearrange(
                    proprioception_input["pred_second_image_z"], "b t s c -> (b t) s c"
                )
                wrist_tokens = self.unpatchify(wrist_tokens)

        else:
            raise NotImplementedError

        if self.predict_wrist_img:
            return wrist_tokens, sampled_token_latent_act
        else:
            return tokens, sampled_token_latent_act


def mar_tiny(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=3,
        encoder_num_heads=6,
        decoder_embed_dim=768,
        decoder_depth=3,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_small(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=768,
        decoder_depth=6,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_embed_dim=1280,
        decoder_depth=20,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
