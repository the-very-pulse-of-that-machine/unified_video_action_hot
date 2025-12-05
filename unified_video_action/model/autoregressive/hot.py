import torch
import torch.nn as nn
from einops import rearrange, repeat


def index_points(points, idx):
    """
    根据索引从点集中选择点
    
    Args:
        points: (B, N, C) 输入点集
        idx: (B, M) 索引
    
    Returns:
        new_points: (B, M, C) 选择的点
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    """
    基于密度峰值聚类和KNN的聚类算法
    
    Args:
        x: (B, N, C) 输入特征
        cluster_num: 聚类数量
        k: KNN的k值
        token_mask: (B, N) token掩码
    
    Returns:
        index_down: (B, cluster_num) 聚类中心索引
        idx_cluster: (B, N) 每个点的聚类标签
    """
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
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # 分配聚类标签
        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        # 确保聚类中心被正确标记
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


class CrossAttention(nn.Module):
    """
    交叉注意力模块，用于从压缩的token中恢复完整序列
    """
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
        """
        Args:
            query: (B, N, C) 查询向量
            key: (B, M, C) 键向量
            value: (B, M, C) 值向量
        
        Returns:
            x: (B, N, C) 注意力输出
        """
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


class HOTModule(nn.Module):
    """
    层次化token压缩模块 (Hierarchical Token Compression)
    
    这个模块可以在transformer的特定层应用token压缩，减少计算复杂度
    同时保持模型性能
    """
    def __init__(self, 
                 token_num=81, 
                 layer_index=3,
                 embed_dim=512,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        
        self.token_num = token_num
        self.layer_index = layer_index
        self.embed_dim = embed_dim
        
        # 聚类相关的组件
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_embed_token = nn.Parameter(torch.zeros(1, self.token_num, embed_dim))
        
        # 交叉注意力用于恢复序列
        self.x_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 可学习的恢复token
        self.cross_attention = CrossAttention(
            embed_dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # 初始化参数
        nn.init.normal_(self.pos_embed_token, std=0.02)
        nn.init.normal_(self.x_token, std=0.02)
    
    def apply_clustering(self, x, original_seq_len):
        """
        应用聚类压缩token
        
        Args:
            x: (B, F, N, C) 输入特征
            original_seq_len: 原始序列长度
        
        Returns:
            compressed_x: (B, token_num, N, C) 压缩后的特征
            index: 聚类中心索引
        """
        B, F, N, C = x.shape
        
        # 计算用于聚类的特征
        x_knn = rearrange(x, 'b f n c -> b (f c) n')
        x_knn = self.pool(x_knn)
        x_knn = rearrange(x_knn, 'b (f c) 1 -> b f c', f=F)
        
        # 应用聚类
        index, idx_cluster = cluster_dpc_knn(x_knn, self.token_num, 2)
        index, _ = torch.sort(index)
        
        # 根据聚类结果选择token
        batch_ind = torch.arange(B, device=x.device).unsqueeze(-1)
        compressed_x = x[batch_ind, index]
        
        # 添加位置编码
        compressed_x = rearrange(compressed_x, 'b f n c -> (b n) f c')
        compressed_x += self.pos_embed_token
        compressed_x = rearrange(compressed_x, '(b n) f c -> b f n c', n=N)
        
        return compressed_x, index
    
    def recover_sequence(self, compressed_x, original_seq_len, num_joints=None):

        # Step 1: 统一转换成 4D
        if compressed_x.ndim == 3:
            if num_joints is None:
                raise ValueError("对于3D输入，必须提供num_joints参数")
            B_N, Fc, C = compressed_x.shape
            B = B_N // num_joints
            compressed_x = rearrange(compressed_x, '(b n) f c -> b f n c', b=B, n=num_joints)

        # Step 2: cross attention 恢复
        B, Fc, N, C = compressed_x.shape
        compressed_x = rearrange(compressed_x, 'b f n c -> (b n) f c')
        x_token = repeat(self.x_token, '() 1 c -> b f c', b=B*N, f=original_seq_len)
        recovered_x = x_token + self.cross_attention(x_token, compressed_x, compressed_x)
        recovered_x = rearrange(recovered_x, '(b n) f c -> b f n c', n=N)

        # Step 3: 返回恢复后的 3D 或 4D 由调用者决定
        return recovered_x

    
    def forward(self, x, current_layer, original_seq_len=None, num_joints=None):
        """
        前向传播
        
        Args:
            x: 输入特征，可以是3D (B*N, F, C) 或 4D (B, F, N, C)
            current_layer: 当前层索引
            original_seq_len: 原始序列长度（如果第一次压缩需要提供）
            num_joints: 关节数量（如果输入是3D需要提供）
        
        Returns:
            x: 处理后的特征
            is_compressed: 是否进行了压缩
        """
        if current_layer == self.layer_index:
            # 在指定层应用压缩
            if original_seq_len is None:
                # 如果没有提供原始序列长度，假设输入是完整序列
                original_seq_len = x.shape[1] if len(x.shape) == 4 else x.shape[0]
            
            # 处理3D输入（来自Transformer层）
            if len(x.shape) == 3:
                if num_joints is None:
                    raise ValueError("对于3D输入，必须提供num_joints参数")
                
                # 将3D转换为4D用于聚类
                B_N, F, C = x.shape
                B = B_N // num_joints
                x_4d = rearrange(x, '(b n) f c -> b f n c', b=B, n=num_joints)
                
                # 应用聚类
                compressed_x_4d, _ = self.apply_clustering(x_4d, original_seq_len)
                
                # 转换回3D
                compressed_x = rearrange(compressed_x_4d, 'b f n c -> (b n) f c')
                return compressed_x, True
            else:
                # 直接处理4D输入
                x, _ = self.apply_clustering(x, original_seq_len)
                return x, True
        
        return x, False


# 使用示例
class HOTTransformer(nn.Module):
    """
    集成了HOT模块的Transformer示例
    """
    def __init__(self, 
                 num_layers=8,
                 embed_dim=512,
                 num_heads=8,
                 mlp_ratio=4,
                 hot_token_num=81,
                 hot_layer_index=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        # HOT模块
        self.hot_module = HOTModule(
            token_num=hot_token_num,
            layer_index=hot_layer_index,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # 普通的transformer层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # 输入嵌入
        self.input_embed = nn.Linear(2, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 243, embed_dim))
        
        # 输出头
        self.output_head = nn.Linear(embed_dim, 3)
    
    def forward(self, x):
        """
        Args:
            x: (B, F, N, 2) 输入序列
        
        Returns:
            output: (B, F, N, 3) 输出序列
        """
        B, F, N, C = x.shape
        
        # 输入嵌入
        x = rearrange(x, 'b f n c -> (b n) f c')
        x = self.input_embed(x)
        x = x + self.pos_embed
        
        original_seq_len = F
        is_compressed = False
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            if not is_compressed:
                x, is_compressed = self.hot_module(
                    x, 
                    i,
                    original_seq_len,
                    num_joints=N     # <-- 这里要加
                )
            x = layer(x)

            
            # Transformer层
            x = layer(x)
        
        # 如果进行了压缩，恢复序列
        if is_compressed:
            x = self.hot_module.recover_sequence(x, original_seq_len, num_joints=N)
        
        # 输出
        x = self.output_head(x)
        x = rearrange(x, '(b n) f c -> b f n c', n=N)
        
        return x


if __name__ == '__main__':
    # 测试HOT模块
    hot_module = HOTModule(token_num=81, layer_index=3, embed_dim=512)
    
    # 测试输入
    batch_size = 2
    seq_len = 243
    num_joints = 17
    embed_dim = 512
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, num_joints, embed_dim)
    
    # 测试压缩
    compressed_x, index = hot_module.apply_clustering(x, seq_len)
    print(f"原始形状: {x.shape}")
    print(f"压缩后形状: {compressed_x.shape}")
    print(f"聚类索引形状: {index.shape}")
    
    # 测试恢复
    recovered_x = hot_module.recover_sequence(compressed_x, seq_len)
    print(f"恢复后形状: {recovered_x.shape}")
    
    # 测试完整HOTTransformer
    model = HOTTransformer(
        num_layers=8,
        embed_dim=512,
        num_heads=8,
        hot_token_num=81,
        hot_layer_index=3
    )
    
    # 模拟2D姿态输入
    input_2d = torch.randn(2, 243, 17, 2)
    output = model(input_2d)
    print(output)
