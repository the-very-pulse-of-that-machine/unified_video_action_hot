#!/usr/bin/env python3
"""
测试mar_con_unified_hot的前向传播 - 按照官方版本的方式
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops import rearrange

# 导入MAR模型
from unified_video_action.model.autoregressive.mar_con_unified_e_h_ttt import mar_base

def test_forward_simple():
    """简单测试MAR模型的前向传播"""
    print("开始测试MAR模型前向传播（官方版本）...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数配置 - 使用与官方配置相同的参数
    kwargs = {
        "task_name": "pusht",  # 使用pusht任务，因为它最简单
        "different_history_freq": False,
        "use_history_action": False,
        "action_mask_ratio": 0.5,
        "use_proprioception": False,
        "predict_wrist_img": False,
        "predict_proprioception": False,
        "shape_meta": {
            "action": {
                "shape": [2]  # pusht任务的动作维度是2
            }
        },
        "language_emb_model": None  # 不使用语言模型
    }
    
    # 创建小模型用于测试
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
        action_model_params={
            "predict_action": True,
            "act_model_type": "conv_fc"
        },
        hot_select_ratio=0.125,
        hot_layer_index=3,
        **kwargs
    )
    
    model = model.to(device)
    model.eval()
    print("模型创建成功")
    
    # 创建模拟输入数据
    # 注意：模型期望的是VAE编码后的潜在表示
    # 对于256x256图像，vae_stride=16，patch_size=1
    # 那么潜在表示的空间尺寸是：256/16=16，所以是16x16的网格
    # vae_embed_dim=16，所以通道数是16
    
    batch_size = 2
    n_frames = 4  # 模型固定使用4帧
    vae_embed_dim = 16
    latent_height = 16  # 256 / 16
    latent_width = 16   # 256 / 16
    
    # 创建VAE潜在表示
    # 输入视频: [B, T, C, H, W] - 这里C是vae_embed_dim，H和W是潜在空间尺寸
    z = torch.randn(batch_size, n_frames, vae_embed_dim, latent_height, latent_width).to(device)
    
    # 条件视频: 同样尺寸
    c = torch.randn(batch_size, n_frames, vae_embed_dim, latent_height, latent_width).to(device)
    
    # 历史动作: 根据代码，当use_history_action=False时，可以为None
    history_nactions = None
    
    # 当前动作: 根据代码，对于video_model任务，nactions可以是None
    nactions = None
    
    # 文本潜在表示: 不使用
    text_latents = None
    
    # 任务模式 - 使用video_model，因为它最简单
    task_mode = "video_model"
    
    # 本体感知输入
    proprioception_input = {}
    
    print(f"输入数据形状:")
    print(f"  z (输入视频潜在表示): {z.shape}")
    print(f"  c (条件视频潜在表示): {c.shape}")
    print(f"  task_mode: {task_mode}")
    
    # 前向传播
    print("\n开始前向传播...")
    try:
        with torch.no_grad():
            loss, video_loss, act_loss = model(
                imgs=z,  # 注意：参数名是imgs，但实际上是VAE潜在表示
                cond=c,
                history_nactions=history_nactions,
                nactions=nactions,
                text_latents=text_latents,
                task_mode=task_mode,
                proprioception_input=proprioception_input
            )
        
        print("前向传播成功!")
        print(f"总损失: {loss.item():.6f}")
        print(f"视频损失: {video_loss.item():.6f}")
        print(f"动作损失: {act_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_tokens_simple():
    """测试sample_tokens方法"""
    print("\n\n测试sample_tokens方法...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一个小模型用于测试
    kwargs = {
        "task_name": "pusht",
        "different_history_freq": False,
        "use_history_action": False,
        "action_mask_ratio": 0.5,
        "use_proprioception": False,
        "predict_wrist_img": False,
        "predict_proprioception": False,
        "shape_meta": {
            "action": {
                "shape": [2]
            }
        },
        "language_emb_model": None
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
        action_model_params={
            "predict_action": True,
            "act_model_type": "conv_fc"
        },
        hot_select_ratio=0.3,
        hot_layer_index=3,
        **kwargs
    )
    
    model = model.to(device)
    model.eval()
    
    # 创建模拟输入
    batch_size = 1
    n_frames = 4
    vae_embed_dim = 16
    latent_height = 16
    latent_width = 16
    
    # 条件输入：VAE潜在表示
    cond = torch.randn(batch_size, n_frames, vae_embed_dim, latent_height, latent_width).to(device)
    
    print(f"条件输入形状: {cond.shape}")
    
    try:
        with torch.no_grad():
            # 测试policy_model模式，因为它会返回动作
            tokens, sampled_actions = model.sample_tokens(
                bsz=batch_size,
                cond=cond,
                text_latents=None,
                num_iter=2,  # 减少迭代次数以加快测试
                cfg=1.0,
                cfg_schedule="linear",
                temperature=1.0,
                progress=False,
                task_mode="policy_model",  # 使用policy_model，它会返回动作
                vae_model=None,
                x=None
            )
        
        print("sample_tokens成功!")
        if tokens is not None:
            print(f"生成的tokens形状: {tokens.shape}")
        if sampled_actions is not None:
            print(f"采样的动作形状: {sampled_actions.shape}")
        
        return True
        
    except Exception as e:
        print(f"sample_tokens失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("测试mar_con_unified_hot前向传播（官方版本）")
    print("=" * 60)
    
    # 测试前向传播
    success1 = test_forward_simple()
    
    # 测试sample_tokens
    success2 = test_sample_tokens_simple()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("所有测试通过!")
    else:
        print("部分测试失败")
    print("=" * 60)
