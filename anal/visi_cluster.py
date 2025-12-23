import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import colorsys
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
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # 分配聚类标签
        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        # 确保聚类中心被正确标记
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down.cpu().numpy(), idx_cluster.cpu().numpy()

def generate_distinct_colors(n_colors, saturation=0.7, value=0.9):
    """生成n个不同的颜色"""
    colors = []
    for i in range(n_colors):
        hue = i / n_colors  # 色相均匀分布
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

class TokenClusterVisualizer:
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
        
        print(f"=== Token Cluster Visualizer Configuration ===")
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
    
    def visualize_cluster_on_frames(self, raw_images, cluster_labels, step_idx, 
                               save_dir="cluster_on_frames"):
        """
        将聚类结果可视化到图像上，每个聚类用不同颜色
        
        raw_images: [1, 16, 3, 96, 96] 原始16帧
        cluster_labels: [N] 每个token的聚类标签
        step_idx: 步骤索引
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B, T_total, C_img, H, W = raw_images.shape
        
        print(f"\n=== Visualizing Step {step_idx} ===")
        print(f"Raw images: {T_total} frames, {H}x{W}")
        print(f"Cluster labels shape: {cluster_labels.shape}")
        
        # 分析聚类统计
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        print(f"Number of clusters: {n_clusters}")
        
        # 生成不同颜色
        cluster_colors = generate_distinct_colors(n_clusters)
        
        # 统计每个聚类的token数
        cluster_sizes = []
        for cluster_id in unique_clusters:
            size = np.sum(cluster_labels == cluster_id)
            cluster_sizes.append(size)
            print(f"  Cluster {cluster_id}: {size} tokens")
        
        # 假设的条件帧映射
        condition_frame_indices = [0, 5, 10, 15]
        
        # 为每个条件帧创建可视化
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
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Step {step_idx}, Frame {orig_frame_idx}: Token Clustering Visualization\n'
                        f'{n_clusters} clusters, {len(cluster_labels)} total tokens', 
                        fontsize=14)
            
            # 子图1: 原始图像
            ax1 = axes[0]
            ax1.imshow(img_display)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # 子图2: 带聚类标注的图像 - 修改为填充颜色
            ax2 = axes[1]
            
            # 创建带透明度的彩色覆盖层
            overlay = np.ones((H, W, 4), dtype=np.float32)  # RGBA格式
            
            # 计算单元格大小
            cell_h = H // self.grid_h
            cell_w = W // self.grid_w
            
            # 绘制所有属于该帧的token
            tokens_in_frame = []
            
            # 先绘制所有单元格的填充颜色
            for token_idx in range(len(cluster_labels)):
                frame_idx, row, col = self.token_to_frame_and_position(token_idx)
                if frame_idx == i:  # 属于当前条件帧
                    cluster_id = cluster_labels[token_idx]
                    color = cluster_colors[cluster_id]
                    
                    x_start = col * cell_w
                    y_start = row * cell_h
                    x_end = min(x_start + cell_w, W)
                    y_end = min(y_start + cell_h, H)
                    
                    # 将颜色添加到覆盖层（带透明度）
                    overlay[y_start:y_end, x_start:x_end, :3] = color  # RGB
                    overlay[y_start:y_end, x_start:x_end, 3] = 0.4     # Alpha (40%透明度)
                    
                    tokens_in_frame.append({
                        'token_idx': token_idx,
                        'cluster_id': cluster_id,
                        'row': row,
                        'col': col,
                        'color': color
                    })
            
            # 显示原始图像
            ax2.imshow(img_display)
            
            # 添加彩色覆盖层
            ax2.imshow(overlay, alpha=0.5)
            
            # 可选：添加细边框来区分单元格
            for token_info in tokens_in_frame:
                x1 = token_info['col'] * cell_w
                y1 = token_info['row'] * cell_h
                
                # 添加细边框
                rect = patches.Rectangle(
                    (x1, y1), cell_w, cell_h,
                    linewidth=0.5, edgecolor='black', facecolor='none', alpha=0.3
                )
                ax2.add_patch(rect)
            
            ax2.set_title(f'Token Clusters ({len(tokens_in_frame)} tokens in this frame)')
            ax2.axis('off')
            
            # 子图3: 聚类中心分布 - 保持原样
            ax3 = axes[2]
            
            # 收集token位置和聚类信息
            rows = [t['row'] for t in tokens_in_frame]
            cols = [t['col'] for t in tokens_in_frame]
            cluster_ids = [t['cluster_id'] for t in tokens_in_frame]
            colors = [t['color'] for t in tokens_in_frame]
            
            if rows and cols:
                # 绘制散点图
                scatter = ax3.scatter(cols, rows, c=colors, s=100, alpha=0.7, 
                                    edgecolors='black', linewidth=1)
                
                # 添加聚类标签
                for idx, token_info in enumerate(tokens_in_frame):
                    ax3.text(token_info['col'], token_info['row'], 
                            str(token_info['cluster_id']),
                            fontsize=8, ha='center', va='center',
                            color='white', fontweight='bold')
                
                ax3.set_xlabel('Column')
                ax3.set_ylabel('Row')
                ax3.set_title('Cluster Distribution (Numbers = Cluster ID)')
                ax3.set_xlim(-0.5, self.grid_w - 0.5)
                ax3.set_ylim(self.grid_h - 0.5, -0.5)  # 反转y轴
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            frame_save_path = os.path.join(save_dir, f'step_{step_idx:03d}_frame_{orig_frame_idx:02d}_clusters.png')
            plt.savefig(frame_save_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved frame {orig_frame_idx}: {frame_save_path}")
        
        # 创建聚类统计图
        self.create_cluster_statistics(cluster_labels, step_idx, save_dir)
        
        return n_clusters, cluster_sizes
        
    def create_cluster_statistics(self, cluster_labels, step_idx, save_dir):
        """创建聚类统计信息"""
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # 统计每个聚类的token数
        cluster_sizes = []
        for cluster_id in unique_clusters:
            size = np.sum(cluster_labels == cluster_id)
            cluster_sizes.append(size)
        
        # 分析空间分布
        spatial_distributions = []
        for cluster_id in unique_clusters:
            # 获取该聚类的所有token索引
            token_indices = np.where(cluster_labels == cluster_id)[0]
            
            # 转换为空间位置
            positions = []
            for token_idx in token_indices:
                frame_idx, row, col = self.token_to_frame_and_position(token_idx)
                positions.append([frame_idx, row, col])
            
            positions = np.array(positions)
            
            # 计算空间统计
            if len(positions) > 0:
                frame_std = positions[:, 0].std() if len(positions) > 1 else 0
                row_std = positions[:, 1].std() if len(positions) > 1 else 0
                col_std = positions[:, 2].std() if len(positions) > 1 else 0
                
                spatial_distributions.append({
                    'cluster_id': cluster_id,
                    'size': len(positions),
                    'frame_std': frame_std,
                    'row_std': row_std,
                    'col_std': col_std,
                    'positions': positions
                })
        
        # 创建统计图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Step {step_idx}: Cluster Statistics', fontsize=16)
        
        # 1. 聚类大小分布
        ax1 = axes[0, 0]
        bars = ax1.bar(range(n_clusters), cluster_sizes, 
                      color=generate_distinct_colors(n_clusters), alpha=0.7)
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Tokens')
        ax1.set_title('Cluster Sizes')
        ax1.set_xticks(range(n_clusters))
        ax1.set_xticklabels([str(i) for i in unique_clusters])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}', ha='center', va='bottom')
        
        # 2. 聚类大小的累积分布
        ax2 = axes[0, 1]
        sorted_sizes = np.sort(cluster_sizes)[::-1]
        cumulative = np.cumsum(sorted_sizes)
        ax2.plot(range(n_clusters), cumulative, 'b-o', linewidth=2)
        ax2.fill_between(range(n_clusters), 0, cumulative, alpha=0.3)
        ax2.set_xlabel('Cluster Rank (sorted by size)')
        ax2.set_ylabel('Cumulative Tokens')
        ax2.set_title('Cumulative Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 帧分布
        ax3 = axes[0, 2]
        frame_counts = np.zeros(self.n_frames)
        for dist in spatial_distributions:
            if len(dist['positions']) > 0:
                frame_indices = dist['positions'][:, 0].astype(int)
                for frame_idx in frame_indices:
                    if frame_idx < self.n_frames:
                        frame_counts[frame_idx] += 1
        
        ax3.bar(range(self.n_frames), frame_counts, alpha=0.7, color='green')
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('Total Tokens')
        ax3.set_title('Token Distribution Across Frames')
        ax3.set_xticks(range(self.n_frames))
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 空间分布散点图（所有聚类）
        ax4 = axes[1, 0]
        all_rows, all_cols, all_colors = [], [], []
        
        for dist in spatial_distributions:
            if len(dist['positions']) > 0:
                rows = dist['positions'][:, 1]
                cols = dist['positions'][:, 2]
                color = generate_distinct_colors(1)[0]  # 每个聚类一个颜色
                
                all_rows.extend(rows)
                all_cols.extend(cols)
                all_colors.extend([color] * len(rows))
        
        if all_rows:
            scatter = ax4.scatter(all_cols, all_rows, c=all_colors, s=50, alpha=0.6)
            ax4.set_xlabel('Column')
            ax4.set_ylabel('Row')
            ax4.set_title('All Clusters Spatial Distribution')
            ax4.set_xlim(-0.5, self.grid_w - 0.5)
            ax4.set_ylim(self.grid_h - 0.5, -0.5)
            ax4.grid(True, alpha=0.3)
        
        # 5. 空间离散度统计
        ax5 = axes[1, 1]
        if spatial_distributions:
            cluster_ids = [d['cluster_id'] for d in spatial_distributions]
            spatial_spreads = []
            
            for dist in spatial_distributions:
                if len(dist['positions']) > 1:
                    # 计算平均最近邻距离作为空间离散度
                    positions_2d = dist['positions'][:, 1:]  # 只取row和col
                    if len(positions_2d) > 1:
                        distances = cdist(positions_2d, positions_2d)
                        np.fill_diagonal(distances, np.inf)
                        min_distances = distances.min(axis=1)
                        spread = min_distances.mean()
                        spatial_spreads.append(spread)
                    else:
                        spatial_spreads.append(0)
                else:
                    spatial_spreads.append(0)
            
            bars5 = ax5.bar(range(len(cluster_ids)), spatial_spreads, 
                           color=generate_distinct_colors(len(cluster_ids)), alpha=0.7)
            ax5.set_xlabel('Cluster ID')
            ax5.set_ylabel('Avg Nearest Neighbor Distance')
            ax5.set_title('Spatial Spread of Clusters')
            ax5.set_xticks(range(len(cluster_ids)))
            ax5.set_xticklabels([str(cid) for cid in cluster_ids])
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 聚类大小与空间离散度的关系
        ax6 = axes[1, 2]
        if spatial_distributions:
            sizes = [d['size'] for d in spatial_distributions]
            spreads = []
            for dist in spatial_distributions:
                if len(dist['positions']) > 1:
                    positions_2d = dist['positions'][:, 1:]
                    if len(positions_2d) > 1:
                        distances = cdist(positions_2d, positions_2d)
                        np.fill_diagonal(distances, np.inf)
                        min_distances = distances.min(axis=1)
                        spread = min_distances.mean()
                        spreads.append(spread)
                    else:
                        spreads.append(0)
                else:
                    spreads.append(0)
            
            scatter6 = ax6.scatter(sizes, spreads, s=100, alpha=0.7, 
                                  c=generate_distinct_colors(len(sizes)))
            ax6.set_xlabel('Cluster Size')
            ax6.set_ylabel('Spatial Spread')
            ax6.set_title('Size vs Spread')
            ax6.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(sizes) > 1:
                z = np.polyfit(sizes, spreads, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(sizes), max(sizes), 100)
                ax6.plot(x_range, p(x_range), "r--", alpha=0.5, label='Trend')
                ax6.legend()
        
        plt.tight_layout()
        stats_save_path = os.path.join(save_dir, f'step_{step_idx:03d}_cluster_statistics.png')
        plt.savefig(stats_save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        # 保存详细统计信息
        stats_text_path = os.path.join(save_dir, f'step_{step_idx:03d}_cluster_details.txt')
        with open(stats_text_path, 'w') as f:
            f.write(f"=== Step {step_idx} Cluster Analysis ===\n")
            f.write(f"Total tokens: {len(cluster_labels)}\n")
            f.write(f"Number of clusters: {n_clusters}\n")
            f.write(f"\nCluster Details:\n")
            
            for i, cluster_id in enumerate(unique_clusters):
                size = cluster_sizes[i]
                percentage = size / len(cluster_labels) * 100
                
                f.write(f"\nCluster {cluster_id}:\n")
                f.write(f"  Size: {size} tokens ({percentage:.1f}%)\n")
                
                # 找出该聚类的token
                token_indices = np.where(cluster_labels == cluster_id)[0]
                
                # 分析帧分布
                frame_dist = {}
                for token_idx in token_indices:
                    frame_idx, _, _ = self.token_to_frame_and_position(token_idx)
                    frame_dist[frame_idx] = frame_dist.get(frame_idx, 0) + 1
                
                f.write(f"  Frame distribution:\n")
                for frame_idx in sorted(frame_dist.keys()):
                    f.write(f"    Frame {frame_idx}: {frame_dist[frame_idx]} tokens\n")
                
                # 分析空间位置
                if token_indices.size > 0:
                    rows, cols = [], []
                    for token_idx in token_indices:
                        _, row, col = self.token_to_frame_and_position(token_idx)
                        rows.append(row)
                        cols.append(col)
                    
                    f.write(f"  Spatial statistics:\n")
                    f.write(f"    Row range: {min(rows)}-{max(rows)}\n")
                    f.write(f"    Col range: {min(cols)}-{max(cols)}\n")
                    
                    if len(rows) > 1:
                        row_mean = np.mean(rows)
                        col_mean = np.mean(cols)
                        row_std = np.std(rows)
                        col_std = np.std(cols)
                        
                        f.write(f"    Row mean±std: {row_mean:.1f}±{row_std:.1f}\n")
                        f.write(f"    Col mean±std: {col_mean:.1f}±{col_std:.1f}\n")
        
        print(f"Cluster statistics saved: {stats_save_path}")
        print(f"Cluster details saved: {stats_text_path}")

def process_pkl_with_cluster_visualization(pkl_path, output_dir="cluster_visualization", 
                                         max_steps=10, cluster_num=100, k=5):
    """
    处理pkl文件，执行聚类并将结果可视化到图像上
    
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
    
    if 'raw_hot_input_token' not in data:
        print("Error: 'raw_hot_input_token' not found in data!")
        return
    
    # 初始化可视化器
    visualizer = TokenClusterVisualizer(
        n_condition_frames=4,
        tokens_per_frame=256,
        grid_size=16,
        original_image_size=(96, 96)
    )
    
    # 确定处理步数
    n_steps = min(len(data["raw_images"]), 
                  len(data["raw_hot_input_token"]),
                  max_steps)
    
    print(f"\nProcessing {n_steps} steps...")
    
    all_cluster_stats = []
    
    for step_idx in range(n_steps):
        print(f"\n{'='*50}")
        print(f"Processing Step {step_idx}")
        print(f"{'='*50}")
        
        # 获取数据
        raw_images = data["raw_images"][step_idx]  # [1, 16, 3, 96, 96]
        hot_input_token = data["raw_hot_input_token"][step_idx]  # [B, N, C]
        
        print(f"Raw images shape: {raw_images.shape}")
        print(f"Hot input token shape: {hot_input_token.shape}")
        
        # 转换hot_input_token为tensor
        if isinstance(hot_input_token, np.ndarray):
            hot_input_tensor = torch.from_numpy(hot_input_token).float()
        else:
            hot_input_tensor = hot_input_token.float()
        
        # 执行聚类
        print(f"Running DPC-KNN clustering (k={k}, clusters={cluster_num})...")
        cluster_indices, cluster_labels = cluster_dpc_knn(
            hot_input_tensor, 
            cluster_num=cluster_num, 
            k=k
        )
        
        # cluster_labels形状: [B, N]，我们取第一个batch
        if len(cluster_labels.shape) > 1:
            cluster_labels = cluster_labels[0]  # [N]
        
        print(f"Cluster labels shape: {cluster_labels.shape}")
        print(f"Unique clusters: {np.unique(cluster_labels)}")
        
        # 可视化聚类结果
        n_clusters, cluster_sizes = visualizer.visualize_cluster_on_frames(
            raw_images, 
            cluster_labels, 
            step_idx,
            save_dir=output_dir
        )
        
        all_cluster_stats.append({
            'step': step_idx,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'total_tokens': len(cluster_labels)
        })
        
        print(f"Step {step_idx} completed: {n_clusters} clusters")
    
    # 创建总体统计
    if all_cluster_stats:
        print(f"\n{'='*50}")
        print(f"Overall Statistics")
        print(f"{'='*50}")
        
        # 计算总体统计
        avg_clusters = np.mean([s['n_clusters'] for s in all_cluster_stats])
        avg_tokens = np.mean([s['total_tokens'] for s in all_cluster_stats])
        
        print(f"Average clusters per step: {avg_clusters:.1f}")
        print(f"Average tokens per step: {avg_tokens:.1f}")
        
        # 分析聚类大小分布
        all_cluster_sizes = []
        for stats in all_cluster_stats:
            all_cluster_sizes.extend(stats['cluster_sizes'])
        
        if all_cluster_sizes:
            print(f"\nCluster Size Statistics:")
            print(f"  Min cluster size: {np.min(all_cluster_sizes)}")
            print(f"  Max cluster size: {np.max(all_cluster_sizes)}")
            print(f"  Mean cluster size: {np.mean(all_cluster_sizes):.1f}")
            print(f"  Median cluster size: {np.median(all_cluster_sizes):.1f}")
            print(f"  Std cluster size: {np.std(all_cluster_sizes):.1f}")
        
        # 保存总体统计
        overall_stats_path = os.path.join(output_dir, "overall_cluster_analysis.txt")
        with open(overall_stats_path, 'w') as f:
            f.write("=== Overall Cluster Analysis ===\n")
            f.write(f"Total steps analyzed: {n_steps}\n")
            f.write(f"Cluster number per step: {cluster_num}\n")
            f.write(f"KNN parameter k: {k}\n")
            f.write(f"Average clusters per step: {avg_clusters:.1f}\n")
            f.write(f"Average tokens per step: {avg_tokens:.1f}\n")
            
            if all_cluster_sizes:
                f.write(f"\nCluster Size Statistics:\n")
                f.write(f"  Min: {np.min(all_cluster_sizes)}\n")
                f.write(f"  Max: {np.max(all_cluster_sizes)}\n")
                f.write(f"  Mean: {np.mean(all_cluster_sizes):.1f}\n")
                f.write(f"  Median: {np.median(all_cluster_sizes):.1f}\n")
                f.write(f"  Std: {np.std(all_cluster_sizes):.1f}\n")
            
            f.write(f"\nStep-by-step Statistics:\n")
            for stats in all_cluster_stats:
                f.write(f"\nStep {stats['step']}:\n")
                f.write(f"  Clusters: {stats['n_clusters']}\n")
                f.write(f"  Tokens: {stats['total_tokens']}\n")
                f.write(f"  Average cluster size: {np.mean(stats['cluster_sizes']):.1f}\n")
        
        print(f"\nAll visualizations saved to: {output_dir}")
        print(f"Overall statistics saved: {overall_stats_path}")
        
        # 创建总体趋势图
        create_overall_trend_plot(all_cluster_stats, output_dir)

def create_overall_trend_plot(all_cluster_stats, output_dir):
    """创建总体趋势图"""
    steps = [s['step'] for s in all_cluster_stats]
    n_clusters = [s['n_clusters'] for s in all_cluster_stats]
    avg_cluster_sizes = [np.mean(s['cluster_sizes']) for s in all_cluster_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cluster Analysis Trends Across Steps', fontsize=16)
    
    # 1. 聚类数量趋势
    ax1 = axes[0, 0]
    ax1.plot(steps, n_clusters, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Step Index')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Number of Clusters per Step')
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均聚类大小趋势
    ax2 = axes[0, 1]
    ax2.plot(steps, avg_cluster_sizes, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Step Index')
    ax2.set_ylabel('Average Cluster Size')
    ax2.set_title('Average Cluster Size per Step')
    ax2.grid(True, alpha=0.3)
    
    # 3. 聚类大小分布箱线图
    ax3 = axes[1, 0]
    cluster_sizes_data = [s['cluster_sizes'] for s in all_cluster_stats]
    box = ax3.boxplot(cluster_sizes_data, patch_artist=True)
    
    # 设置箱线图颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(box['boxes'], colors * (len(cluster_sizes_data) // len(colors) + 1)):
        patch.set_facecolor(color)
    
    ax3.set_xlabel('Step Index')
    ax3.set_ylabel('Cluster Size')
    ax3.set_title('Cluster Size Distribution per Step')
    ax3.set_xticklabels([str(s) for s in steps])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 散点图：聚类数量 vs 平均聚类大小
    ax4 = axes[1, 1]
    scatter = ax4.scatter(n_clusters, avg_cluster_sizes, c=steps, cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Average Cluster Size')
    ax4.set_title('Clusters vs Average Size')
    ax4.grid(True, alpha=0.3)
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax4, label='Step Index')
    
    plt.tight_layout()
    trend_path = os.path.join(output_dir, "overall_cluster_trends.png")
    plt.savefig(trend_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"Overall trend plot saved: {trend_path}")

if __name__ == "__main__":
    # 使用示例
    pkl_path = "3.pkl"  # 修改为您的pkl文件路径
    output_dir = "token_cluster_visualization"
    
    # 参数说明：
    # pkl_path: 输入pkl文件路径
    # output_dir: 输出目录
    # max_steps: 最大处理步数
    # cluster_num: 聚类数量
    # k: KNN参数（通常5-10）
    
    process_pkl_with_cluster_visualization(
        pkl_path=pkl_path,
        output_dir=output_dir,
        max_steps=5,       # 只处理前5个步骤（每个步骤会生成多个图像）
        cluster_num=100,   # 聚类100个中心
        k=5               # KNN使用5个最近邻
    )