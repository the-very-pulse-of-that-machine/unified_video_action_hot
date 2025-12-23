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
        _, index_down = torch.topk(1/score, k=cluster_num, dim=-1)

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

def visualize_single_step_four_subplots(raw_images, cluster_labels, cluster_centers, step_idx, 
                                       output_dir="four_subplot_visualization",
                                       target_frame_idx=0,
                                       grid_size=16,
                                       n_frames=4,
                                       tokens_per_frame=256):
    """
    为单个步骤生成一张包含四个子图的图片，标出聚类中心
    1. 原始图像
    2. 带聚类颜色覆盖的图像
    3. 聚类大小分布条形图
    4. 聚类空间分布散点图（标出中心）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像信息
    B, T_total, C_img, H, W = raw_images.shape
    
    # 分析聚类统计
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # 统计每个聚类的token数
    cluster_sizes = []
    for cluster_id in unique_clusters:
        size = np.sum(cluster_labels == cluster_id)
        cluster_sizes.append(size)
    
    # 生成颜色
    cluster_colors = generate_distinct_colors(n_clusters)
    
    # 获取目标帧图像
    if target_frame_idx >= T_total:
        target_frame_idx = 0
    
    img = raw_images[0, target_frame_idx]  # [C, H, W]
    
    # 转换图像格式
    if img.max() <= 1.0 and img.min() >= 0:
        img_display = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    elif img.min() >= -1 and img.max() <= 1:
        img_display = ((img + 1) / 2 * 255).transpose(1, 2, 0).astype(np.uint8)
    else:
        img_display = img.transpose(1, 2, 0).astype(np.uint8)
    
    # ========== 关键改动：提取聚类中心信息 ==========
    # cluster_centers 是从 cluster_dpc_knn 返回的 index_down
    center_positions = {}  # 存储聚类中心位置信息
    for cluster_idx, center_token_idx in enumerate(cluster_centers[0]):  # 取第一个batch
        # 将token索引转换为(帧索引, 行, 列)
        frame_idx = center_token_idx // tokens_per_frame
        spatial_idx = center_token_idx % tokens_per_frame
        row = spatial_idx // grid_size
        col = spatial_idx % grid_size
        
        # 找到该中心对应的聚类ID（cluster_labels中对应的值）
        center_cluster_id = cluster_labels[center_token_idx]
        
        center_positions[center_cluster_id] = {
            'token_idx': int(center_token_idx),
            'frame_idx': int(frame_idx),
            'row': int(row),
            'col': int(col),
            'is_in_target_frame': (frame_idx == (target_frame_idx % n_frames))
        }
    # ========== 改动结束 ==========
    
    # 创建包含四个子图的大图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Step {step_idx}: Token Clustering Analysis\nFrame {target_frame_idx}, {n_clusters} Clusters, {len(cluster_labels)} Tokens', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # ========== 子图1: 原始图像 ==========
    ax1 = axes[0, 0]
    ax1.imshow(img_display)
    ax1.set_title('(a) Original Image', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel(f'Frame {target_frame_idx}', fontsize=12)
    ax1.axis('off')
    
    # ========== 子图2: 带聚类颜色覆盖的图像 ==========
    ax2 = axes[0, 1]
    
    # 创建带透明度的彩色覆盖层
    overlay = np.ones((H, W, 4), dtype=np.float32)  # RGBA格式
    
    # 计算单元格大小
    cell_h = H // grid_size
    cell_w = W // grid_size
    
    # 收集属于目标帧的token
    tokens_in_target_frame = []
    
    for token_idx in range(len(cluster_labels)):
        # 将token索引转换为(帧索引, 行, 列)
        frame_idx = token_idx // tokens_per_frame
        spatial_idx = token_idx % tokens_per_frame
        row = spatial_idx // grid_size
        col = spatial_idx % grid_size
        
        if frame_idx == (target_frame_idx % n_frames):  # 属于目标条件帧
            cluster_id = cluster_labels[token_idx]
            color_idx = np.where(unique_clusters == cluster_id)[0][0]
            color = cluster_colors[color_idx]
            
            x_start = col * cell_w
            y_start = row * cell_h
            x_end = min(x_start + cell_w, W)
            y_end = min(y_start + cell_h, H)
            
            # 将颜色添加到覆盖层（带透明度）
            overlay[y_start:y_end, x_start:x_end, :3] = color  # RGB
            overlay[y_start:y_end, x_start:x_end, 3] = 0.5     # Alpha (50%透明度)
            
            # 检查是否为聚类中心
            is_center = (token_idx in cluster_centers[0])
            center_info = None
            if is_center:
                center_info = {
                    'is_center': True,
                    'cluster_id': cluster_id
                }
            
            tokens_in_target_frame.append({
                'token_idx': token_idx,
                'cluster_id': cluster_id,
                'row': row,
                'col': col,
                'color': color,
                'color_idx': color_idx,
                'is_center': is_center,
                'center_info': center_info
            })
    
    # 显示原始图像
    ax2.imshow(img_display)
    
    # 添加彩色覆盖层
    ax2.imshow(overlay, alpha=0.6)
    
    # ========== 关键改动：在覆盖层中标记聚类中心 ==========
    # 用特殊边框标记聚类中心
    for token_info in tokens_in_target_frame:
        x1 = token_info['col'] * cell_w
        y1 = token_info['row'] * cell_h
        
        # 所有单元格都有细边框
        rect = patches.Rectangle(
            (x1, y1), cell_w, cell_h,
            linewidth=0.8, edgecolor='white', facecolor='none', alpha=0.8
        )
        ax2.add_patch(rect)
        
        # 如果是聚类中心，添加特殊标记
        if token_info['is_center']:
            # 添加红色星形标记
            center_x = x1 + cell_w / 2
            center_y = y1 + cell_h / 2
            
            star = patches.RegularPolygon(
                (center_x, center_y),
                numVertices=5,  # 五角星
                radius=cell_w/3,
                orientation=np.pi/10,
                facecolor='red',
                edgecolor='yellow',
                linewidth=2,
                alpha=0.9
            )
            ax2.add_patch(star)
            
            # 添加中心标签
            ax2.text(center_x, center_y - cell_h/4, 'C', 
                    fontsize=12, fontweight='bold', ha='center', va='center',
                    color='yellow', bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8))
    # ========== 改动结束 ==========
    
    ax2.set_title(f'(b) Token Clusters ({len(tokens_in_target_frame)} tokens in frame)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Color indicates cluster membership | Red stars: Cluster Centers', fontsize=12)
    ax2.axis('off')
    
    # ========== 子图3: 聚类大小分布条形图 ==========
    ax3 = axes[1, 0]
    
    # 按大小排序聚类
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    sorted_clusters = unique_clusters[sorted_indices]
    sorted_sizes = np.array(cluster_sizes)[sorted_indices]
    sorted_colors = np.array(cluster_colors)[sorted_indices]
    
    bars = ax3.bar(range(n_clusters), sorted_sizes, 
                  color=sorted_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Cluster ID (sorted by size)', fontsize=13)
    ax3.set_ylabel('Number of Tokens', fontsize=13)
    ax3.set_title('(c) Cluster Size Distribution', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xticks(range(n_clusters))
    ax3.set_xticklabels([str(int(cid)) for cid in sorted_clusters], fontsize=10, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.tick_params(axis='both', labelsize=11)
    
    # 添加数值标签
    for i, (bar, size) in enumerate(zip(bars, sorted_sizes)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{size}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='black')
    
    # 添加统计信息，包含中心信息
    avg_size = np.mean(cluster_sizes)
    max_size = np.max(cluster_sizes)
    min_size = np.min(cluster_sizes)
    
    # ========== 关键改动：在统计信息中显示中心数量 ==========
    centers_in_frame = sum(1 for token in tokens_in_target_frame if token['is_center'])
    total_centers = len(cluster_centers[0])
    
    stats_text = f'Avg: {avg_size:.1f}\nMax: {max_size}\nMin: {min_size}\nCenters: {centers_in_frame}/{total_centers} in frame'
    ax3.text(0.95, 0.95, stats_text,
             transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # ========== 改动结束 ==========
    
    # ========== 子图4: 聚类空间分布散点图（标出中心） ==========
    ax4 = axes[1, 1]
    
    if tokens_in_target_frame:
        # 收集所有token的位置和颜色
        rows = [t['row'] for t in tokens_in_target_frame]
        cols = [t['col'] for t in tokens_in_target_frame]
        color_indices = [t['color_idx'] for t in tokens_in_target_frame]
        cluster_ids = [t['cluster_id'] for t in tokens_in_target_frame]
        is_centers = [t['is_center'] for t in tokens_in_target_frame]
        
        # 绘制散点图 - 普通点
        regular_points = [i for i, is_center in enumerate(is_centers) if not is_center]
        if regular_points:
            ax4.scatter(np.array(cols)[regular_points], np.array(rows)[regular_points], 
                       c=np.array(color_indices)[regular_points], 
                       cmap='tab20c', s=80, alpha=0.6, 
                       edgecolors='black', linewidth=0.5, zorder=1)
        
        # ========== 关键改动：用特殊标记绘制聚类中心 ==========
        center_points = [i for i, is_center in enumerate(is_centers) if is_center]
        if center_points:
            # 使用星形标记聚类中心
            scatter_centers = ax4.scatter(np.array(cols)[center_points], np.array(rows)[center_points], 
                                         c='red', s=400, marker='*',  # 红色星形
                                         edgecolors='yellow', linewidth=2.5,
                                         alpha=1.0, zorder=3,  # 最高zorder确保在最上层
                                         label='Cluster Centers')
            
            # 用三角形标记背景，增强可见性
            ax4.scatter(np.array(cols)[center_points], np.array(rows)[center_points], 
                       c='black', s=450, marker='^',  # 黑色三角形背景
                       alpha=0.3, zorder=2)
        
        # 添加聚类ID标签（只标注较大的聚类和中心）
        for token_info in tokens_in_target_frame:
            # 只标注大小超过平均值的聚类或者是中心
            cluster_size = cluster_sizes[np.where(unique_clusters == token_info['cluster_id'])[0][0]]
            if cluster_size > avg_size or token_info['is_center']:
                label_color = 'white' if token_info['is_center'] else 'black'
                bbox_color = 'red' if token_info['is_center'] else 'black'
                bbox_alpha = 0.7 if token_info['is_center'] else 0.5
                
                ax4.text(token_info['col'], token_info['row'], 
                        str(token_info['cluster_id']),
                        fontsize=10, ha='center', va='center',
                        fontweight='bold', color=label_color,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor=bbox_color, alpha=bbox_alpha))
        
        # 添加图例
        if center_points:
            ax4.legend(handles=[scatter_centers], loc='upper right', fontsize=11)
        
        ax4.set_xlabel('Column (Grid Position)', fontsize=13)
        ax4.set_ylabel('Row (Grid Position)', fontsize=13)
        ax4.set_title('(d) Spatial Distribution (Stars: Cluster Centers)', 
                     fontsize=14, fontweight='bold', pad=10)
        ax4.set_xlim(-0.5, grid_size - 0.5)
        ax4.set_ylim(grid_size - 0.5, -0.5)  # 反转y轴
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.tick_params(axis='both', labelsize=11)
        
        # 添加统计信息
        stats_text = f'Tokens shown: {len(tokens_in_target_frame)}\nCenters: {len(center_points)} in frame'
        ax4.text(0.02, 0.98, stats_text,
                transform=ax4.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No tokens in this frame',
                transform=ax4.transAxes, fontsize=14,
                ha='center', va='center')
        ax4.set_title('(d) Spatial Distribution', fontsize=14, fontweight='bold', pad=10)
        ax4.axis('off')
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f'step_{step_idx:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved visualization with cluster centers: {save_path}")
    
    # 返回统计信息，包含中心信息
    centers_in_target_frame = sum(1 for token in tokens_in_target_frame if token['is_center'])
    
    return {
        'step': step_idx,
        'n_clusters': n_clusters,
        'total_tokens': len(cluster_labels),
        'total_centers': len(cluster_centers[0]),
        'tokens_in_frame': len(tokens_in_target_frame),
        'centers_in_frame': centers_in_target_frame,
        'avg_cluster_size': np.mean(cluster_sizes),
        'max_cluster_size': np.max(cluster_sizes),
        'min_cluster_size': np.min(cluster_sizes),
        'save_path': save_path,
        'center_positions': center_positions
    }

def process_pkl_four_subplot_visualization(pkl_path, output_dir="four_subplot_results", 
                                         max_steps=5, cluster_num=100, k=5,
                                         target_frame_idx=0):
    """
    处理pkl文件，为每个步骤生成一张四子图可视化
    """
    print(f"Loading data from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"\n=== Data Information ===")
    print(f"Raw images steps: {len(data['raw_images'])}")
    print(f"Selected token steps: {len(data['selected_token_indices'])}")
    
    if 'raw_hot_input_token' not in data:
        print("Error: 'raw_hot_input_token' not found in data!")
        return
    
    # 参数配置
    n_frames = 4
    tokens_per_frame = 256
    grid_size = 16
    
    # 确定处理步数
    n_steps = min(len(data["raw_images"]), 
                  len(data["raw_hot_input_token"]),
                  max_steps)
    
    print(f"\nProcessing {n_steps} steps for four-subplot visualization...")
    print(f"Target frame index: {target_frame_idx}")
    print(f"Cluster number: {cluster_num}")
    print(f"Output directory: {output_dir}")
    
    all_stats = []
    
    for step_idx in range(n_steps):
        print(f"\n{'='*60}")
        print(f"Processing Step {step_idx}")
        print(f"{'='*60}")
        
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
        print(f"Number of unique clusters: {len(np.unique(cluster_labels))}")
        
        # 生成四子图可视化
        stats = visualize_single_step_four_subplots(
            raw_images=raw_images,
            cluster_labels=cluster_labels,
            step_idx=step_idx,
            cluster_centers=cluster_indices,
            output_dir=output_dir,
            target_frame_idx=target_frame_idx,
            grid_size=grid_size,
            n_frames=n_frames,
            tokens_per_frame=tokens_per_frame
        )
        
        all_stats.append(stats)
        
        print(f"Step {step_idx} completed:")
        print(f"  Clusters: {stats['n_clusters']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Tokens in frame {target_frame_idx}: {stats['tokens_in_frame']}")
        print(f"  Avg cluster size: {stats['avg_cluster_size']:.1f}")
    
    # 生成总体统计报告
    if all_stats:
        print(f"\n{'='*60}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*60}")
        
        # 计算总体统计
        avg_clusters = np.mean([s['n_clusters'] for s in all_stats])
        avg_tokens = np.mean([s['total_tokens'] for s in all_stats])
        avg_frame_tokens = np.mean([s['tokens_in_frame'] for s in all_stats])
        avg_cluster_size = np.mean([s['avg_cluster_size'] for s in all_stats])
        
        print(f"Steps analyzed: {n_steps}")
        print(f"Average clusters per step: {avg_clusters:.1f}")
        print(f"Average total tokens per step: {avg_tokens:.1f}")
        print(f"Average tokens in frame {target_frame_idx}: {avg_frame_tokens:.1f}")
        print(f"Average cluster size: {avg_cluster_size:.1f}")
        
        # 保存总体统计
        overall_stats_path = os.path.join(output_dir, "overall_statistics.txt")
        with open(overall_stats_path, 'w') as f:
            f.write("=== FOUR-SUBPLOT VISUALIZATION OVERALL STATISTICS ===\n\n")
            f.write(f"Input file: {pkl_path}\n")
            f.write(f"Steps analyzed: {n_steps}\n")
            f.write(f"Target frame index: {target_frame_idx}\n")
            f.write(f"Cluster number per step: {cluster_num}\n")
            f.write(f"KNN parameter k: {k}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Average clusters per step: {avg_clusters:.1f}\n")
            f.write(f"  Average total tokens per step: {avg_tokens:.1f}\n")
            f.write(f"  Average tokens in target frame: {avg_frame_tokens:.1f}\n")
            f.write(f"  Average cluster size: {avg_cluster_size:.1f}\n\n")
            
            f.write("STEP-BY-STEP DETAILS:\n")
            for stats in all_stats:
                f.write(f"\nStep {stats['step']}:\n")
                f.write(f"  Clusters: {stats['n_clusters']}\n")
                f.write(f"  Total tokens: {stats['total_tokens']}\n")
                f.write(f"  Tokens in frame {target_frame_idx}: {stats['tokens_in_frame']}\n")
                f.write(f"  Avg/Max/Min cluster size: {stats['avg_cluster_size']:.1f}/{stats['max_cluster_size']}/{stats['min_cluster_size']}\n")
                f.write(f"  Output file: {stats['save_path']}\n")
        
        print(f"\nAll visualizations saved to: {output_dir}")
        print(f"Overall statistics saved: {overall_stats_path}")
        
        # 创建汇总图表
        create_summary_chart(all_stats, output_dir)
    
    return all_stats

def create_summary_chart(all_stats, output_dir):
    """创建步骤间统计汇总图表"""
    steps = [s['step'] for s in all_stats]
    n_clusters = [s['n_clusters'] for s in all_stats]
    total_tokens = [s['total_tokens'] for s in all_stats]
    frame_tokens = [s['tokens_in_frame'] for s in all_stats]
    avg_sizes = [s['avg_cluster_size'] for s in all_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Summary of Cluster Statistics Across Steps', fontsize=16, fontweight='bold')
    
    # 1. 聚类数量趋势
    ax1 = axes[0, 0]
    ax1.plot(steps, n_clusters, 'b-o', linewidth=2.5, markersize=8, markerfacecolor='white')
    ax1.fill_between(steps, 0, n_clusters, alpha=0.2, color='blue')
    ax1.set_xlabel('Step Index', fontsize=12)
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_title('Clusters per Step', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=11)
    
    # 2. Token数量趋势
    ax2 = axes[0, 1]
    width = 0.35
    x = np.arange(len(steps))
    ax2.bar(x - width/2, total_tokens, width, label='Total Tokens', alpha=0.7, color='green')
    ax2.bar(x + width/2, frame_tokens, width, label='Tokens in Target Frame', alpha=0.7, color='orange')
    ax2.set_xlabel('Step Index', fontsize=12)
    ax2.set_ylabel('Number of Tokens', fontsize=12)
    ax2.set_title('Token Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in steps])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend(fontsize=11)
    ax2.tick_params(axis='both', labelsize=11)
    
    # 3. 平均聚类大小
    ax3 = axes[1, 0]
    bars = ax3.bar(steps, avg_sizes, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Step Index', fontsize=12)
    ax3.set_ylabel('Average Cluster Size', fontsize=12)
    ax3.set_title('Average Cluster Size per Step', fontsize=13, fontweight='bold')
    ax3.set_xticks(steps)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.tick_params(axis='both', labelsize=11)
    
    # 添加数值标签
    for bar, size in zip(bars, avg_sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 4. 聚类数量 vs 平均聚类大小散点图
    ax4 = axes[1, 1]
    scatter = ax4.scatter(n_clusters, avg_sizes, c=steps, 
                         cmap='viridis', s=150, alpha=0.8, 
                         edgecolors='black', linewidth=1.5)
    ax4.set_xlabel('Number of Clusters', fontsize=12)
    ax4.set_ylabel('Average Cluster Size', fontsize=12)
    ax4.set_title('Clusters vs Average Size', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.tick_params(axis='both', labelsize=11)
    
    # 添加趋势线
    if len(n_clusters) > 1:
        z = np.polyfit(n_clusters, avg_sizes, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(n_clusters), max(n_clusters), 100)
        ax4.plot(x_range, p(x_range), "r--", alpha=0.7, linewidth=2, 
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        ax4.legend(fontsize=11)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Step Index', fontsize=11)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "summary_across_steps.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Summary chart saved: {summary_path}")

if __name__ == "__main__":
    # 使用示例
    pkl_path = "3.pkl"  # 修改为您的pkl文件路径
    output_dir = "cluster_visualization"
    
    # 参数说明：
    # pkl_path: 输入pkl文件路径
    # output_dir: 输出目录
    # max_steps: 最大处理步数（建议5-10步，每步生成一张图）
    # cluster_num: 聚类数量
    # k: KNN参数（通常5）
    # target_frame_idx: 要可视化的目标帧索引（0, 5, 10, 15对应4个条件帧）
    
    process_pkl_four_subplot_visualization(
        pkl_path=pkl_path,
        output_dir=output_dir,
        max_steps=10,             # 处理前5个步骤
        cluster_num=300,         # 聚类100个中心
        k=2,                    # KNN使用5个最近邻
        target_frame_idx=0       # 可视化第0帧（可改为5, 10, 15查看不同帧）
    )