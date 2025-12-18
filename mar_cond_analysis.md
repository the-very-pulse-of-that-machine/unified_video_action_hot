# MAR模型中的cond参数分析

## 1. cond的基本定义

在MAR（Masked Autoencoder）模型中，`cond`是**条件输入**（conditional input），通常表示**观测图像**或**条件图像**，用于指导模型生成或预测。

## 2. cond的数据结构

### 原始输入形状
- **形状**: `[B, T, C, H, W]`
  - `B`: 批量大小（batch size）
  - `T`: 时间步数（通常是4帧）
  - `C`: 通道数（3，RGB）
  - `H, W`: 高度和宽度（通常是256x256）

### 处理后的形状
经过`patchify`处理后：
- **形状**: `[B, T, S, C']`
  - `S`: patch数量（`seq_len = seq_h * seq_w`）
  - `C'`: patch特征维度（`token_embed_dim = vae_embed_dim * patch_size²`）
  - 对于默认参数：`seq_h = seq_w = 16`，`seq_len = 256`，`vae_embed_dim = 16`，`patch_size = 1`，所以`C' = 16`

## 3. cond的处理流程

### 步骤1: Patchify（分块）
```python
# 原始cond形状: [B, T, C, H, W]
cond = rearrange(cond, "b t c h w -> (b t) c h w")
cond = self.patchify(cond)  # 转换为[B*T, S, C']
cond = rearrange(cond, "(b t) seq_len c -> b t seq_len c", b=B)
```

### 步骤2: 投影到编码器维度
```python
cond = self.z_proj_cond(cond)  # 线性投影: [B, T, S, C'] -> [B, T, S, encoder_embed_dim]
cond = rearrange(cond, "b t s c -> b (t s) c")  # 展平: [B, T*S, encoder_embed_dim]
```

### 步骤3: 根据任务模式处理
不同的任务模式对`cond`的处理不同：

#### a) `policy_model`（策略模型）
- `cond`: 作为条件输入
- `x`: 使用`fake_latent_x`替代原始输入
- 用于动作预测任务

#### b) `inverse_model`（逆模型）
- `cond`: 使用`fake_latent_x`替代
- `x`: 使用原始输入
- 用于从动作反推状态

#### c) `video_model` / `dynamic_model` / `full_dynamic_model`（视频模型）
- `cond`: 作为条件输入
- `x`: 使用原始输入，但masked部分用`fake_latent_x`替换
- 用于视频生成任务

### 步骤4: 与其他模态拼接
`cond`会与其他模态特征拼接：
```python
# 基本拼接
parts = [x, cond]  # x是输入，cond是条件

# 可选添加的模态
if self.use_history_action:
    parts.append(history_action_latents_expand)
parts.append(action_latents_expand)

if self.use_proprioception:
    if self.task_name == "umi":
        parts.append(proprioception_state_cond_expand)
    else:
        parts.extend([proprioception_image_cond, proprioception_state_cond_expand])

# 最终拼接
x = torch.cat(parts, dim=-1)
```

### 步骤5: 模态融合
```python
x = self.proj_cond_x_layer(x)  # 投影到统一维度
```

### 步骤6: 添加位置编码
```python
# 时空位置编码
temporal_pos_embed_expanded = self.temporal_pos_embed.unsqueeze(2).expand(-1, -1, S, -1)
spatial_pos_embed_expanded = self.spatial_pos_embed.unsqueeze(1).expand(-1, T, -1, -1)
combined_pos_embed = (temporal_pos_embed_expanded + spatial_pos_embed_expanded).reshape(-1, T*S, embed_dim)

x = x + combined_pos_embed  # 添加位置编码
```

## 4. cond在不同任务模式中的作用

### 4.1 视频生成任务（video_model）
- **作用**: 提供上下文信息，指导视频生成
- **处理**: 作为条件输入，与masked输入一起编码
- **示例**: 给定前几帧，生成后续帧

### 4.2 策略学习任务（policy_model）
- **作用**: 提供当前状态观测
- **处理**: 作为主要输入，用于预测动作
- **示例**: 给定当前图像，预测机器人动作

### 4.3 逆模型任务（inverse_model）
- **作用**: 作为目标状态
- **处理**: 使用fake latent替代，实际输入是x
