import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from matplotlib.lines import Line2D  # 添加这行
from scipy.spatial.distance import cdist

def index_points(points, idx):
    """Helper function to index points"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def cluster_dpc_knn(x, cluster_num, k=5, token_mask=None):
    """
    DPC-KNN clustering algorithm
    x: [B, N, C] tensor
    cluster_num: number of clusters
    k: number of nearest neighbors
    token_mask: optional mask for valid tokens
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
        _, index_down = torch.topk(1/score, k=cluster_num, dim=-1)

        # 分配聚类标签
        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        # 确保聚类中心被正确标记
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down.cpu().numpy(), idx_cluster.cpu().numpy()

def select_channel(x, select_ratio=0.3, temporal_ratio=0.5, token_mask=None, num_frames=4):
    with torch.no_grad():
        B, N, C = x.shape
        num_channels = N // num_frames
        
        device = x.device
        
        # 重塑为 [B, num_frames, num_channels, C]
        x_reshaped = x.view(B, num_frames, num_channels, C)
        
        # 计算连续帧对的距离
        frame_i = x_reshaped[:, :-1, :, :]
        frame_j = x_reshaped[:, 1:, :, :]
        distances = torch.norm(frame_i - frame_j, dim=-1)
        
        # 取每个通道的最大距离
        max_distance_per_channel, _ = torch.max(distances, dim=1)
        
        # 处理掩码
        if token_mask is not None:
            if token_mask.dim() == 2:  # [B, 1024]
                mask_reshaped = token_mask.view(B, num_frames, num_channels)
            else:  # [B, 4, 256]
                mask_reshaped = token_mask
            
            # 通道在所有帧中至少一帧有效
            valid_channels = mask_reshaped.any(dim=1)  # [B, 256]
            max_distance_per_channel = max_distance_per_channel.masked_fill(~valid_channels, -float('inf'))
        
        # 选择距离最大的通道
        select_num = max(1, int(num_channels * select_ratio * temporal_ratio))

        _, selected_channels = torch.topk(max_distance_per_channel, k=select_num, dim=-1)  # [B, select_num]
        
        # 生成所有帧的索引
        frame_indices = torch.arange(num_frames, device=device).view(1, 1, num_frames)
        channel_indices_expanded = selected_channels.unsqueeze(-1).expand(-1, -1, num_frames)
        batch_indices = frame_indices * num_channels + channel_indices_expanded
        
        index_down = batch_indices.reshape(B, -1)  # [B, select_num * num_frames]

        # 调用聚类函数
        cluster_indices2, cluster_labels = cluster_dpc_knn(
            x, 
            cluster_num=int(N*select_ratio*temporal_ratio), 
            k=2
        )
        
        # 确保cluster_indices2是tensor并移到正确设备
        if isinstance(cluster_indices2, np.ndarray):
            cluster_indices2 = torch.from_numpy(cluster_indices2).to(device)
        
        # 如果cluster_indices2是一维的，可能是聚类中心索引，为每个批次复制
        if cluster_indices2.dim() == 1:
            cluster_indices2 = cluster_indices2.unsqueeze(0).expand(B, -1)
        elif cluster_indices2.dim() == 2 and cluster_indices2.shape[0] == 1:
            # 如果是二维但只有一批，扩展到所有批次
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
    

class ClusterVisualizer:
    def __init__(self, 
                 n_condition_frames=4,
                 tokens_per_frame=256,
                 grid_size=16,
                 original_image_size=(96, 96)):
        """
        初始化可视化器
        """
        self.n_frames = n_condition_frames      # 4
        self.tokens_per_frame = tokens_per_frame  # 256
        self.grid_h = self.grid_w = grid_size   # 16
        self.total_tokens = n_condition_frames * tokens_per_frame  # 1024
        
        self.orig_h, self.orig_w = original_image_size  # 96, 96
        
        print(f"=== Cluster Visualizer Configuration ===")
        print(f"Condition frames: {self.n_frames}")
        print(f"Tokens per frame: {self.tokens_per_frame} ({self.grid_h}x{self.grid_w} grid)")
        print(f"Total tokens: {self.total_tokens}")
        print(f"Original image: {self.orig_h}x{self.orig_w}")
    
    def token_to_frame_and_position(self, token_idx):
        """将token索引转换为(帧索引, 行, 列)"""
        frame_idx = token_idx // self.tokens_per_frame
        spatial_idx = token_idx % self.tokens_per_frame
        row = spatial_idx // self.grid_w
        col = spatial_idx % self.grid_w
        return frame_idx, row, col
    
    def visualize_cluster_results(self, raw_images, hot_input_token, step_idx, 
                                cluster_indices, original_indices, save_dir="cluster_visualizations"):
        """
        可视化聚类结果
        
        raw_images: [1, 16, 3, 96, 96] 原始16帧
        hot_input_token: [B, N, C] 热点输入token
        step_idx: 步骤索引
        cluster_indices: 聚类选择的indices 
            新格式：长度为B的列表，每个元素是批次的索引数组（或单个numpy数组，B=1时）
        original_indices: 原始模型选择的indices [n_selected]
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B, T_total, C_img, H, W = raw_images.shape
        B_token, N, C_token = hot_input_token.shape
        
        print(f"\n=== Visualizing Step {step_idx} ===")
        print(f"Raw images shape: {raw_images.shape}")
        print(f"Hot input token shape: {hot_input_token.shape}")
        
        # 处理cluster_indices的格式（适配新的select_channel返回格式）
        if isinstance(cluster_indices, list):
            # 新的select_channel返回格式：列表，每个元素是一个批次的索引
            print(f"cluster_indices is list with {len(cluster_indices)} batches")
            
            # 检查列表中的元素类型
            batch_results = []
            for i, batch_cluster in enumerate(cluster_indices):
                if isinstance(batch_cluster, np.ndarray):
                    batch_results.append(batch_cluster)
                elif isinstance(batch_cluster, torch.Tensor):
                    batch_results.append(batch_cluster.cpu().numpy())
                else:
                    # 尝试转换为numpy数组
                    batch_results.append(np.array(batch_cluster))
            
            # 默认使用第一个批次的索引（假设batch_size=1或我们只关心第一个样本）
            if len(batch_results) > 0:
                cluster_indices_array = batch_results[0]
                print(f"Using first batch with {len(cluster_indices_array)} indices")
            else:
                cluster_indices_array = np.array([])
                print("Warning: Empty cluster_indices list")
        elif isinstance(cluster_indices, np.ndarray):
            # 如果是numpy数组，直接使用（兼容旧格式）
            cluster_indices_array = cluster_indices
            print(f"cluster_indices is numpy array with shape {cluster_indices.shape}")
        elif isinstance(cluster_indices, torch.Tensor):
            # 如果是torch tensor，转换为numpy
            cluster_indices_array = cluster_indices.cpu().numpy()
            print(f"cluster_indices is torch tensor, converted to numpy array with shape {cluster_indices_array.shape}")
        else:
            # 其他类型，尝试转换
            try:
                cluster_indices_array = np.array(cluster_indices)
                print(f"cluster_indices converted to numpy array with shape {cluster_indices_array.shape}")
            except:
                print(f"Error: Cannot convert cluster_indices type {type(cluster_indices)}")
                cluster_indices_array = np.array([])
        
        # 确保cluster_indices_array是一维数组
        if cluster_indices_array.ndim > 1:
            cluster_indices_array = cluster_indices_array.flatten()
        
        print(f"Cluster selected {len(cluster_indices_array)} tokens")
        print(f"Original model selected {len(original_indices)} tokens")
        
        # 分析聚类选择的分布
        cluster_frame_dist = {}
        for token_idx in cluster_indices_array:
            # 确保token_idx是标量
            token_idx = int(token_idx)
            frame_idx, row, col = self.token_to_frame_and_position(token_idx)
            if frame_idx not in cluster_frame_dist:
                cluster_frame_dist[frame_idx] = []
            cluster_frame_dist[frame_idx].append((row, col))
        
        # 分析原始选择的分布
        original_frame_dist = {}
        for token_idx in original_indices:
            # 确保token_idx是标量
            token_idx = int(token_idx)
            frame_idx, row, col = self.token_to_frame_and_position(token_idx)
            if frame_idx not in original_frame_dist:
                original_frame_dist[frame_idx] = []
            original_frame_dist[frame_idx].append((row, col))
        
        # 假设的条件帧映射
        condition_frame_indices = [0, 5, 10, 15]
        
        # 创建对比可视化
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f'Step {step_idx}: Clustering vs Original Selection\n'
                    f'Cluster: {len(cluster_indices_array)} | Original: {len(original_indices)}',
                    fontsize=16)
        
        for i, orig_frame_idx in enumerate(condition_frame_indices):
            if orig_frame_idx >= T_total:
                continue
                
            # 获取原始图像
            img = raw_images[0, orig_frame_idx]  # [C, H, W]
            
            # 转换图像格式
            if img.max() <= 1.0 and img.min() >= 0:
                img_display = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            elif img.min() >= -1 and img.max() <= 1:
                img_display = ((img + 1) / 2 * 255).transpose(1, 2, 0).astype(np.uint8)
            else:
                img_display = img.transpose(1, 2, 0).astype(np.uint8)
            
            # 子图1: 原始图像
            ax1 = axes[i, 0]
            ax1.imshow(img_display)
            ax1.set_title(f'Frame {orig_frame_idx}')
            ax1.axis('off')
            
            # 子图2: 聚类选择结果
            ax2 = axes[i, 1]
            ax2.imshow(img_display)
            
            # 绘制聚类选择的tokens
            if i in cluster_frame_dist:
                for row, col in cluster_frame_dist[i]:
                    cell_h = H // self.grid_h
                    cell_w = W // self.grid_w
                    x1 = col * cell_w
                    y1 = row * cell_h
                    
                    rect = patches.Rectangle(
                        (x1, y1), cell_w, cell_h,
                        linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7,
                        label='Cluster' if row==0 and col==0 else ""
                    )
                    ax2.add_patch(rect)
            
            ax2.set_title(f'Cluster Selection\n{len(cluster_frame_dist.get(i, []))} tokens')
            ax2.axis('off')
            
            # 子图3: 原始模型选择结果
            ax3 = axes[i, 2]
            ax3.imshow(img_display)
            
            # 绘制原始模型选择的tokens
            if i in original_frame_dist:
                for row, col in original_frame_dist[i]:
                    cell_h = H // self.grid_h
                    cell_w = W // self.grid_w
                    x1 = col * cell_w
                    y1 = row * cell_h
                    
                    rect = patches.Rectangle(
                        (x1, y1), cell_w, cell_h,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7,
                        label='Original' if row==0 and col==0 else ""
                    )
                    ax3.add_patch(rect)
            
            ax3.set_title(f'Original Selection\n{len(original_frame_dist.get(i, []))} tokens')
            ax3.axis('off')
            
            # 子图4: 两者对比
            ax4 = axes[i, 3]
            ax4.imshow(img_display)
            
            # 绘制聚类选择（蓝色）
            if i in cluster_frame_dist:
                for row, col in cluster_frame_dist[i]:
                    cell_h = H // self.grid_h
                    cell_w = W // self.grid_w
                    x1 = col * cell_w
                    y1 = row * cell_h
                    
                    rect = patches.Rectangle(
                        (x1, y1), cell_w, cell_h,
                        linewidth=2, edgecolor='blue', facecolor='none', alpha=0.5
                    )
                    ax4.add_patch(rect)
            
            # 绘制原始选择（红色）
            if i in original_frame_dist:
                for row, col in original_frame_dist[i]:
                    cell_h = H // self.grid_h
                    cell_w = W // self.grid_w
                    x1 = col * cell_w
                    y1 = row * cell_h
                    
                    rect = patches.Rectangle(
                        (x1, y1), cell_w, cell_h,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.5,
                        linestyle='--'
                    )
                    ax4.add_patch(rect)
            
            # 计算重叠
            overlap_count = 0
            if i in cluster_frame_dist and i in original_frame_dist:
                cluster_set = set(cluster_frame_dist[i])
                original_set = set(original_frame_dist[i])
                overlap_set = cluster_set.intersection(original_set)
                overlap_count = len(overlap_set)
                
                # 绘制重叠区域（紫色）
                for row, col in overlap_set:
                    cell_h = H // self.grid_h
                    cell_w = W // self.grid_w
                    x1 = col * cell_w
                    y1 = row * cell_h
                    
                    rect = patches.Rectangle(
                        (x1, y1), cell_w, cell_h,
                        linewidth=3, edgecolor='purple', facecolor='none', alpha=0.8
                    )
                    ax4.add_patch(rect)
            
            ax4.set_title(f'Comparison\nOverlap: {overlap_count}')
            ax4.axis('off')
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Cluster Selection'),
            Line2D([0], [0], color='red', lw=2, label='Original Selection'),
            Line2D([0], [0], color='purple', lw=3, label='Overlap'),
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'step_{step_idx:03d}_cluster_vs_original.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster visualization saved: {save_path}")
        
        # 创建统计分析（使用处理后的数组）
        self.create_cluster_analysis(cluster_indices_array, original_indices, step_idx, save_dir)
        
        return overlap_count
    
    def create_cluster_analysis(self, cluster_indices, original_indices, step_idx, save_dir):
        """创建聚类统计分析"""
        # 确保输入是可迭代的Python列表（处理新格式）
        if isinstance(cluster_indices, list):
            # 如果是列表，可能是多批次结果，取第一个批次
            if len(cluster_indices) > 0:
                if isinstance(cluster_indices[0], np.ndarray):
                    cluster_indices = cluster_indices[0].tolist()
                elif isinstance(cluster_indices[0], torch.Tensor):
                    cluster_indices = cluster_indices[0].cpu().numpy().tolist()
                else:
                    cluster_indices = list(cluster_indices[0])
            else:
                cluster_indices = []
        elif isinstance(cluster_indices, np.ndarray):
            cluster_indices = cluster_indices.tolist()
        
        if isinstance(original_indices, np.ndarray):
            original_indices = original_indices.tolist()
        elif isinstance(original_indices, torch.Tensor):
            original_indices = original_indices.cpu().numpy().tolist()
        
        # 计算重叠统计
        cluster_set = set(cluster_indices)
        original_set = set(original_indices)
        overlap_set = cluster_set.intersection(original_set)
        
        total_cluster = len(cluster_indices)
        total_original = len(original_indices)
        total_overlap = len(overlap_set)
        
        # 计算重叠率
        overlap_rate_cluster = total_overlap / total_cluster if total_cluster > 0 else 0
        overlap_rate_original = total_overlap / total_original if total_original > 0 else 0
        
        # 分析帧分布
        cluster_frames = {}
        original_frames = {}
        
        for token_idx in cluster_indices:
            token_idx = int(token_idx)  # 确保是整数
            frame_idx, _, _ = self.token_to_frame_and_position(token_idx)
            cluster_frames[frame_idx] = cluster_frames.get(frame_idx, 0) + 1
        
        for token_idx in original_indices:
            token_idx = int(token_idx)  # 确保是整数
            frame_idx, _, _ = self.token_to_frame_and_position(token_idx)
            original_frames[frame_idx] = original_frames.get(frame_idx, 0) + 1
        
        # 创建分析图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Step {step_idx}: Cluster Analysis', fontsize=14)
        
        # 1. 选择数量对比
        ax1 = axes[0, 0]
        categories = ['Cluster', 'Original', 'Overlap']
        values = [total_cluster, total_original, total_overlap]
        colors = ['blue', 'red', 'purple']
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Token Count')
        ax1.set_title('Selection Count Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}', ha='center', va='bottom')
        
        # 2. 重叠率
        ax2 = axes[0, 1]
        rates = [overlap_rate_cluster * 100, overlap_rate_original * 100]
        rate_labels = ['Overlap/Cluster', 'Overlap/Original']
        bars2 = ax2.bar(rate_labels, rates, color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('Overlap Rate (%)')
        ax2.set_title('Overlap Rates')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars2, rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. 帧分布对比
        ax3 = axes[1, 0]
        frames = sorted(set(list(cluster_frames.keys()) + list(original_frames.keys())))
        cluster_counts = [cluster_frames.get(f, 0) for f in frames]
        original_counts = [original_frames.get(f, 0) for f in frames]
        
        x = np.arange(len(frames))
        width = 0.35
        ax3.bar(x - width/2, cluster_counts, width, label='Cluster', color='blue', alpha=0.7)
        ax3.bar(x + width/2, original_counts, width, label='Original', color='red', alpha=0.7)
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('Token Count')
        ax3.set_title('Distribution Across Frames')
        ax3.set_xticks(x)
        ax3.set_xticklabels(frames)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 空间分布散点图
        ax4 = axes[1, 1]
        
        # 收集所有点的位置
        all_rows, all_cols, colors, sizes = [], [], [], []
        
        for token_idx in cluster_indices:
            token_idx = int(token_idx)
            _, row, col = self.token_to_frame_and_position(token_idx)
            all_rows.append(row)
            all_cols.append(col)
            colors.append('blue' if token_idx not in overlap_set else 'purple')
            sizes.append(50 if token_idx not in overlap_set else 80)
        
        for token_idx in original_indices:
            token_idx = int(token_idx)
            if token_idx not in cluster_set:  # 只添加不在cluster中的点
                _, row, col = self.token_to_frame_and_position(token_idx)
                all_rows.append(row)
                all_cols.append(col)
                colors.append('red')
                sizes.append(50)
        
        if all_rows:
            scatter = ax4.scatter(all_cols, all_rows, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax4.set_xlabel('Column')
            ax4.set_ylabel('Row')
            ax4.set_title('Spatial Distribution')
            ax4.set_xlim(-0.5, self.grid_w-0.5)
            ax4.set_ylim(self.grid_h-0.5, -0.5)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'step_{step_idx:03d}_cluster_analysis.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        # 保存统计信息
        stats_path = os.path.join(save_dir, f'step_{step_idx:03d}_cluster_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"=== Cluster Analysis for Step {step_idx} ===\n")
            f.write(f"Cluster selected tokens: {total_cluster}\n")
            f.write(f"Original selected tokens: {total_original}\n")
            f.write(f"Overlap tokens: {total_overlap}\n")
            f.write(f"Overlap/Cluster rate: {overlap_rate_cluster:.2%}\n")
            f.write(f"Overlap/Original rate: {overlap_rate_original:.2%}\n")
            f.write(f"\nFrame distribution (Cluster):\n")
            for frame in sorted(cluster_frames.keys()):
                f.write(f"  Frame {frame}: {cluster_frames[frame]} tokens\n")
            f.write(f"\nFrame distribution (Original):\n")
            for frame in sorted(original_frames.keys()):
                f.write(f"  Frame {frame}: {original_frames[frame]} tokens\n")
        
        print(f"Cluster analysis saved: {save_path}")
        print(f"Cluster stats saved: {stats_path}")


def process_pkl_file(pkl_path, output_dir="cluster_results", max_steps=10, cluster_num=307, k=5):
    """
    处理pkl文件，执行聚类分析
    
    pkl_path: pkl文件路径
    output_dir: 输出目录
    max_steps: 最大处理步数
    cluster_num: 聚类数量
    k: KNN参数
    """
    import pickle
    
    print(f"Loading data from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"\n=== Data Information ===")
    print(f"Raw images steps: {len(data['raw_images'])}")
    print(f"Selected token steps: {len(data['selected_token_indices'])}")
    print(f"Hot input token steps: {len(data['raw_hot_input_token'])}")
    
    if 'raw_hot_input_token' not in data:
        print("Error: 'raw_hot_input_token' not found in data!")
        return
    
    # 初始化可视化器
    visualizer = ClusterVisualizer(
        n_condition_frames=4,
        tokens_per_frame=256,
        grid_size=16,
        original_image_size=(96, 96)
    )
    
    # 确定处理步数
    n_steps = min(len(data["raw_images"]), 
                  len(data["selected_token_indices"]),
                  len(data["raw_hot_input_token"]),
                  max_steps)
    
    print(f"\nProcessing {n_steps} steps...")
    
    total_overlaps = []
    
    for step_idx in range(n_steps):
        print(f"\n--- Processing Step {step_idx} ---")
        
        # 获取数据
        raw_images = data["raw_images"][step_idx]  # [1, 16, 3, 96, 96]
        hot_input_token = data["raw_hot_input_token"][step_idx]  # [B, N, C]
        original_indices = data["selected_token_indices"][step_idx].flatten()  # [n_selected]
        
        print(f"Hot input token shape: {hot_input_token.shape}")
        print(f"Original indices shape: {original_indices.shape}")
        
        # 转换hot_input_token为tensor
        if isinstance(hot_input_token, np.ndarray):
            hot_input_tensor = torch.from_numpy(hot_input_token).float()
        else:
            hot_input_tensor = hot_input_token.float()
        
        # 执行聚类（使用新的select_channel函数）
        print(f"Running DPC-KNN clustering (k={k}, clusters={cluster_num})...")

        cluster_indices = select_channel(hot_input_tensor, select_ratio=0.3)
        
        # 打印聚类结果信息
        if isinstance(cluster_indices, list):
            print(f"select_channel returned list with {len(cluster_indices)} batches")
            for i, batch_indices in enumerate(cluster_indices):
                if hasattr(batch_indices, 'shape'):
                    print(f"  Batch {i}: {batch_indices.shape} indices")
                else:
                    print(f"  Batch {i}: {len(batch_indices)} indices")
        else:
            print(f"select_channel returned: {type(cluster_indices)}")
            if hasattr(cluster_indices, 'shape'):
                print(f"Shape: {cluster_indices.shape}")
        
        # 可视化结果
        overlap_count = visualizer.visualize_cluster_results(
            raw_images, 
            hot_input_token, 
            step_idx,
            cluster_indices,
            original_indices,
            save_dir=output_dir
        )
        
        total_overlaps.append(overlap_count)
    
    # 创建总体统计
    if total_overlaps:
        print(f"\n=== Overall Statistics ===")
        print(f"Average overlap per step: {np.mean(total_overlaps):.1f} ± {np.std(total_overlaps):.1f}")
        print(f"Min overlap: {np.min(total_overlaps)}")
        print(f"Max overlap: {np.max(total_overlaps)}")
        
        # 保存总体统计
        overall_stats_path = os.path.join(output_dir, "overall_cluster_stats.txt")
        with open(overall_stats_path, 'w') as f:
            f.write("=== Overall Cluster Analysis ===\n")
            f.write(f"Total steps analyzed: {n_steps}\n")
            f.write(f"Cluster number: {cluster_num}\n")
            f.write(f"KNN parameter k: {k}\n")
            f.write(f"Average overlap per step: {np.mean(total_overlaps):.1f}\n")
            f.write(f"Std deviation: {np.std(total_overlaps):.1f}\n")
            f.write(f"Min overlap: {np.min(total_overlaps)}\n")
            f.write(f"Max overlap: {np.max(total_overlaps)}\n")
            f.write(f"\nOverlap per step:\n")
            for i, overlap in enumerate(total_overlaps):
                f.write(f"  Step {i}: {overlap} tokens\n")
        
        print(f"\nAll visualizations and analyses saved to: {output_dir}")
        print(f"Overall statistics saved: {overall_stats_path}")


if __name__ == "__main__":
    # 使用示例
    pkl_path = "3.pkl"  # 修改为您的pkl文件路径
    output_dir = "cluster_analysis_results_0_2"
    
    # 参数说明：
    # pkl_path: 输入pkl文件路径
    # output_dir: 输出目录
    # max_steps: 最大处理步数（避免处理太多）
    # cluster_num: 聚类数量（与原模型选择的token数量相近）
    # k: KNN参数（通常5-10）
    
    process_pkl_file(
        pkl_path=pkl_path,
        output_dir=output_dir,
        max_steps=10,      # 只处理前10个步骤
        cluster_num=205,   # 聚类100个中心（与原模型选择数量相近）
        k=5               # KNN使用5个最近邻
    )