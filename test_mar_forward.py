#!/usr/bin/env python3
"""
测试mar_con_unified_hot的前向传播 - 添加文本latent
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops import rearrange

# 导入MAR模型
from unified_video_action.model.autoregressive.mar_con_unified_expand_hot import mar_base

def test_forward_simple():
    """简单测试MAR模型的前向传播"""
    print("开始测试MAR模型前向传播（官方版本）...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数
    kwargs = {
        "task_name": "pusht",
        "different_history_freq": False,
        "use_history_action": False,
        "action_mask_ratio": 0.5,
        "use_proprioception": False,
        "predict_wrist_img": False,
        "predict_proprioception": False,
        "shape_meta": {
            "action": {"shape": [2]}
        },
        "language_emb_model": "clip",  # 使用文本
        "language_emb_model_type": 1,  # type 1
        "buffer_size_text": 4           # 假设文本 token 数为4
    }
    
    model = mar_base(
        img_size=256,
        vae_stride=16,
        patch_size=1,
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
        action_model_params={"predict_action": True, "act_model_type": "conv_fc"},
        hot_select_ratio=0.125,
        hot_layer_index=3,
        **kwargs
    ).to(device)
    
    model.eval()
    print("模型创建成功")
    
    # 输入视频 latent
    batch_size = 2
    n_frames = 4
    vae_embed_dim = 16
    latent_h = latent_w = 16
    z = torch.randn(batch_size, n_frames, vae_embed_dim, latent_h, latent_w).to(device)
    c = torch.randn(batch_size, n_frames, vae_embed_dim, latent_h, latent_w).to(device)
    
    # 文本 latent: [B, buffer_size_text, C]
    text_latents = torch.randn(batch_size, 256, 768).to(device)
    
    print(f"输入数据形状:")
    print(f"  z: {z.shape}, c: {c.shape}, text_latents: {text_latents.shape}")
    
    # 前向传播
    print("\n开始前向传播...")
    try:
        with torch.no_grad():
            loss, video_loss, act_loss = model(
                imgs=z,
                cond=c,
                history_nactions=None,
                nactions=None,
                text_latents=text_latents,  # 传入文本latent
                task_mode="video_model",
                proprioception_input={}
            )
        print("前向传播成功!")
        print(f"总损失: {loss.item():.6f}, 视频损失: {video_loss.item():.6f}, 动作损失: {act_loss.item():.6f}")
        return True
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sample_tokens_simple():
    """测试sample_tokens方法，添加文本latent"""
    print("\n测试sample_tokens方法...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kwargs = {
        "task_name": "pusht",
        "different_history_freq": False,
        "use_history_action": False,
        "action_mask_ratio": 0.5,
        "use_proprioception": False,
        "predict_wrist_img": False,
        "predict_proprioception": False,
        "shape_meta": {"action": {"shape": [2]}},
        "language_emb_model": "clip",
        "language_emb_model_type": 1,
        "buffer_size_text": 4
    }
    
    model = mar_base(
        img_size=256,
        vae_stride=16,
        patch_size=1,
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
        action_model_params={"predict_action": True, "act_model_type": "conv_fc"},
        hot_select_ratio=0.3,
        hot_layer_index=3,
        **kwargs
    ).to(device)
    model.eval()
    
    batch_size = 1
    n_frames = 4
    vae_embed_dim = 16
    latent_h = latent_w = 16
    
    cond = torch.randn(batch_size, n_frames, vae_embed_dim, latent_h, latent_w).to(device)
    text_latents = torch.randn(batch_size, kwargs["buffer_size_text"], vae_embed_dim).to(device)
    
    try:
        with torch.no_grad():
            tokens, sampled_actions = model.sample_tokens(
                bsz=batch_size,
                cond=cond,
                text_latents=text_latents,  # 传入文本latent
                num_iter=2,
                cfg=1.0,
                cfg_schedule="linear",
                temperature=1.0,
                progress=False,
                task_mode="policy_model",
                vae_model=None,
                x=None
            )
        print("sample_tokens成功!")
        if tokens is not None:
            print(f"生成的tokens形状: {tokens.shape}")
        if sampled_actions is not None:
            print(f"采样动作形状: {sampled_actions.shape}")
        return True
    except Exception as e:
        print(f"sample_tokens失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("测试mar_con_unified_hot前向传播（添加text latent）")
    print("=" * 60)
    
    success1 = test_forward_simple()
    success2 = test_sample_tokens_simple()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("所有测试通过!")
    else:
        print("部分测试失败")
    print("=" * 60)
