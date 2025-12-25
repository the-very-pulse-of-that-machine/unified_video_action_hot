import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Type
import numpy as np


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k=2, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        # 计算距离矩阵
        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        # 应用token掩码
        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # 计算KNN距离
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        # 计算密度
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        # 计算父节点和距离
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # 计算得分并选择聚类中心
        score = dist * density
        
        #_, index_down = torch.topk(score, k=cluster_num, dim=-1)
        _, index_down = torch.topk(1/score, k=cluster_num, dim=-1)

        # 分配聚类标签
        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        # 确保聚类中心被正确标记
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


def select_channel(x, select_ratio=0.3, temporal_ratio=0.5, token_mask=None, num_frames=4):
    with torch.no_grad():
        B, N, C = x.shape
        num_channels = N // num_frames
        device = x.device
        x_reshaped = x.view(B, num_frames, num_channels, C)
        
        frame_i = x_reshaped[:, :-1, :, :]
        frame_j = x_reshaped[:, 1:, :, :]
        distances = torch.norm(frame_i - frame_j, dim=-1)
        
        max_distance_per_channel, _ = torch.max(distances, dim=1)
        
        if token_mask is not None:
            if token_mask.dim() == 2:  # [B, 1024]
                mask_reshaped = token_mask.view(B, num_frames, num_channels)
            else:  # [B, 4, 256]
                mask_reshaped = token_mask
            
            valid_channels = mask_reshaped.any(dim=1)  # [B, 256]
            max_distance_per_channel = max_distance_per_channel.masked_fill(~valid_channels, -float('inf'))
        select_num = max(1, int(num_channels * select_ratio * temporal_ratio))
        _, selected_channels = torch.topk(max_distance_per_channel, k=select_num, dim=-1)  # [B, select_num]
        frame_indices = torch.arange(num_frames, device=device).view(1, 1, num_frames)
        channel_indices_expanded = selected_channels.unsqueeze(-1).expand(-1, -1, num_frames)
        batch_indices = frame_indices * num_channels + channel_indices_expanded
        
        index_down = batch_indices.reshape(B, -1)  # [B, select_num * num_frames]
        cluster_indices2, cluster_labels = cluster_dpc_knn(
            x, 
            cluster_num=int(N*select_ratio*temporal_ratio), 
            k=2
        )
        
        if isinstance(cluster_indices2, np.ndarray):
            cluster_indices2 = torch.from_numpy(cluster_indices2).to(device)
        
        if cluster_indices2.dim() == 1:
            cluster_indices2 = cluster_indices2.unsqueeze(0).expand(B, -1)
        elif cluster_indices2.dim() == 2 and cluster_indices2.shape[0] == 1:
            cluster_indices2 = cluster_indices2.expand(B, -1)
        
        batch_selected_indices = []

        target_num = int(N * select_ratio)

        for i in range(B):
            combined = torch.cat([
                index_down[i].flatten(),
                cluster_indices2[i].flatten()
            ])

            unique_indices = torch.unique(combined)
            current_num = unique_indices.numel()

            if current_num < target_num:
                needed_num = target_num - current_num
                all_token_indices = torch.arange(N, device=unique_indices.device)

                mask = ~torch.isin(all_token_indices, unique_indices)
                available_indices = all_token_indices[mask]
                torch.manual_seed(i)

                perm = torch.randperm(available_indices.numel(), device=unique_indices.device)
                additional_indices = available_indices[perm[:needed_num]]
                final_indices = torch.cat([unique_indices, additional_indices])
            else:
                final_indices = unique_indices

            batch_selected_indices.append(final_indices)
        batch_selected_indices = torch.stack(batch_selected_indices)

        return batch_selected_indices

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B, N, C = query.shape
        B, M, C = value.shape

        q = self.linear_q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP module with GELU activation"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        bias: bool = True,
        drop: float = 0.,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class LayerScale(nn.Module):
    """Layer scale module"""
    
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma