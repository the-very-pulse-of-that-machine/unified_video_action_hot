import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

class TokenVisualizer:
    def __init__(self, 
                 n_condition_frames=4,
                 tokens_per_frame=256,
                 grid_size=16,
                 original_image_size=(96, 96),
                 model_image_size=256):
        """
        Initialize with all confirmed information
        """
        self.n_frames = n_condition_frames      # 4
        self.tokens_per_frame = tokens_per_frame  # 256
        self.grid_h = self.grid_w = grid_size   # 16
        self.total_tokens = n_condition_frames * tokens_per_frame  # 1024
        
        self.orig_h, self.orig_w = original_image_size  # 96, 96
        self.model_h = self.model_w = model_image_size  # 256, 256
        
        # Calculate scaling factors
        self.scale_h = self.model_h // self.grid_h  # 256÷16=16
        self.scale_w = self.model_w // self.grid_w  # 256÷16=16
        
        # Original image to model image scaling ratio
        self.orig_to_model_h = self.model_h / self.orig_h  # ≈2.67
        self.orig_to_model_w = self.model_w / self.orig_w  # ≈2.67
        
        print(f"=== Token Visualizer Configuration ===")
        print(f"Condition frames: {self.n_frames}")
        print(f"Tokens per frame: {self.tokens_per_frame} ({self.grid_h}x{self.grid_w} grid)")
        print(f"Total tokens: {self.total_tokens}")
        print(f"Original image: {self.orig_h}x{self.orig_w}")
        print(f"Model internal processing: {self.model_h}x{self.model_w}")
        print(f"Each token corresponds to: {self.scale_h}x{self.scale_w} pixels (in {self.model_h}x{self.model_w} image)")
        print(f"Scaling ratio: {self.orig_to_model_h:.2f}x")
    
    def token_to_frame_and_position(self, token_idx):
        """Convert token index to (frame_index, row, column)"""
        frame_idx = token_idx // self.tokens_per_frame
        spatial_idx = token_idx % self.tokens_per_frame
        row = spatial_idx // self.grid_w
        col = spatial_idx % self.grid_w
        return frame_idx, row, col
    
    def visualize_single_step(self, raw_images, token_indices, step_idx, save_dir="step_visualizations"):
        """
        Visualize token selection for one inference step
        
        raw_images: [1, 16, 3, 96, 96] Original 16 frames
        token_indices: Selected token indices [n_selected_tokens]
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B, T_total, C, H, W = raw_images.shape
        
        print(f"\n=== Visualizing Step {step_idx} ===")
        print(f"Original images: {T_total} frames, size: {H}x{W}")
        print(f"Selected tokens: {len(token_indices.flatten())}")
        
        # Analyze token distribution for this step
        tokens = token_indices.flatten()
        frame_distribution = {}
        for token_idx in tokens:
            frame_idx, row, col = self.token_to_frame_and_position(token_idx)
            if frame_idx not in frame_distribution:
                frame_distribution[frame_idx] = []
            frame_distribution[frame_idx].append((row, col))
        
        print(f"Tokens distributed across {len(frame_distribution)} condition frames:")
        for frame_idx in sorted(frame_distribution.keys()):
            print(f"  Frame {frame_idx}: {len(frame_distribution[frame_idx])} tokens")
        
        # Create visualization for this step
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f'Step {step_idx}: Token Visualization\n'
                    f'Selected {len(tokens)} tokens from {self.total_tokens} total',
                    fontsize=16)
        
        # Assumed condition frame mapping
        condition_frame_indices = [0, 5, 10, 15]  # Assuming uniform sampling from 16 frames
        
        for i, orig_frame_idx in enumerate(condition_frame_indices):
            if orig_frame_idx >= T_total:
                continue
                
            # Get original image
            img = raw_images[0, orig_frame_idx]  # [C, H, W]
            
            # Convert to display format
            if img.max() <= 1.0 and img.min() >= 0:
                img_display = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            elif img.min() >= -1 and img.max() <= 1:
                img_display = ((img + 1) / 2 * 255).transpose(1, 2, 0).astype(np.uint8)
            else:
                img_display = img.transpose(1, 2, 0).astype(np.uint8)
            
            # Show original image
            ax = axes[i, 0]
            ax.imshow(img_display)
            ax.set_title(f'Original Frame {orig_frame_idx}\n(Assumed condition frame {i})')
            ax.axis('off')
            
            # Show image with token grid
            ax = axes[i, 1]
            ax.imshow(img_display)
            
            # Draw grid
            for y in range(0, H, H//self.grid_h):
                ax.axhline(y, color='black', alpha=0.3, linewidth=0.5)
            for x in range(0, W, W//self.grid_w):
                ax.axvline(x, color='black', alpha=0.3, linewidth=0.5)
            
            ax.set_title(f'16×16 Grid\nEach cell = 1 token')
            ax.axis('off')
            
            # Show selected tokens for this condition frame
            ax = axes[i, 2]
            ax.imshow(img_display)
            
            if i in frame_distribution:
                for row, col in frame_distribution[i]:
                    # Calculate position in original image
                    cell_h = H // self.grid_h
                    cell_w = W // self.grid_w
                    x1 = col * cell_w
                    y1 = row * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    
                    rect = patches.Rectangle(
                        (x1, y1), cell_w, cell_h,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                    )
                    ax.add_patch(rect)
            
            ax.set_title(f'Selected tokens: {len(frame_distribution.get(i, []))}')
            ax.axis('off')
            
            # Show token heatmap for this frame
            ax = axes[i, 3]
            heatmap = np.zeros((self.grid_h, self.grid_w))
            if i in frame_distribution:
                for row, col in frame_distribution[i]:
                    heatmap[row, col] = 1.0
            
            im = ax.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Token Selection Heatmap')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'step_{step_idx:03d}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"Step visualization saved: {save_path}")
        
        # Create analysis for this single step
        #self.create_single_step_analysis(tokens, step_idx, save_dir)
        
        return frame_distribution
    
    def create_single_step_analysis(self, tokens, step_idx, save_dir):
        """Create analysis for a single step"""
        # Analyze spatial distribution
        all_rows, all_cols = [], []
        frame_counts = []
        
        for token_idx in tokens:
            frame_idx, row, col = self.token_to_frame_and_position(token_idx)
            all_rows.append(row)
            all_cols.append(col)
            frame_counts.append(frame_idx)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Step {step_idx}: Token Selection Analysis', fontsize=14)
        
        # 1. Frame distribution
        axes[0, 0].hist(frame_counts, bins=range(self.n_frames+1), 
                       alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Condition Frame Index')
        axes[0, 0].set_ylabel('Token Count')
        axes[0, 0].set_title(f'Token Distribution Across Condition Frames')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Row distribution
        axes[0, 1].hist(all_rows, bins=range(self.grid_h+1),
                       alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Row Index')
        axes[0, 1].set_ylabel('Token Count')
        axes[0, 1].set_title(f'Token Distribution Across Rows')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Column distribution
        axes[1, 0].hist(all_cols, bins=range(self.grid_w+1),
                       alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Column Index')
        axes[1, 0].set_ylabel('Token Count')
        axes[1, 0].set_title(f'Token Distribution Across Columns')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Spatial scatter plot
        if all_rows and all_cols:
            scatter = axes[1, 1].scatter(all_cols, all_rows, 
                                        c=frame_counts, cmap='viridis',
                                        alpha=0.6, s=50)
            axes[1, 1].set_xlabel('Column Index')
            axes[1, 1].set_ylabel('Row Index')
            axes[1, 1].set_title('Token Spatial Distribution (Color=Frame Index)')
            axes[1, 1].set_xlim(-0.5, self.grid_w-0.5)
            axes[1, 1].set_ylim(self.grid_h-0.5, -0.5)  # Invert y-axis
            plt.colorbar(scatter, ax=axes[1, 1], label='Frame Index')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'step_{step_idx:03d}_analysis.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    
    def create_aggregate_statistics(self, data, output_dir="aggregate_statistics"):
        """
        Create aggregate statistics and visualizations for all steps together
        
        data: Dictionary with 'raw_images' and 'selected_token_indices'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        raw_images = data["raw_images"]
        token_indices_list = data["selected_token_indices"]
        
        n_steps = min(len(raw_images), len(token_indices_list))
        
        print(f"\n=== Creating Aggregate Statistics ===")
        print(f"Total steps to analyze: {n_steps}")
        
        # =================================================================
        # 1. COLLECT ALL DATA FOR AGGREGATE ANALYSIS
        # =================================================================
        all_tokens = []
        all_frame_indices = []
        all_rows = []
        all_cols = []
        tokens_per_step = []
        
        for step_idx in range(n_steps):
            tokens = token_indices_list[step_idx].flatten()
            tokens_per_step.append(len(tokens))
            
            for token_idx in tokens:
                frame_idx, row, col = self.token_to_frame_and_position(token_idx)
                all_tokens.append(token_idx)
                all_frame_indices.append(frame_idx)
                all_rows.append(row)
                all_cols.append(col)
        
        total_selections = len(all_tokens)
        
        print(f"\n=== Aggregate Statistics ===")
        print(f"Total token selections across all steps: {total_selections}")
        print(f"Average tokens per step: {np.mean(tokens_per_step):.1f} ± {np.std(tokens_per_step):.1f}")
        print(f"Min tokens per step: {np.min(tokens_per_step)}")
        print(f"Max tokens per step: {np.max(tokens_per_step)}")
        
        # =================================================================
        # 2. CREATE AGGREGATE HEATMAPS
        # =================================================================
        print(f"\nCreating aggregate heatmaps...")
        
        # 2.1 Overall token frequency heatmap (all frames combined)
        overall_heatmap = np.zeros((self.grid_h, self.grid_w))
        
        # 2.2 Per-frame heatmaps
        frame_heatmaps = np.zeros((self.n_frames, self.grid_h, self.grid_w))
        
        for step_idx in range(n_steps):
            tokens = token_indices_list[step_idx].flatten()
            for token_idx in tokens:
                frame_idx, row, col = self.token_to_frame_and_position(token_idx)
                overall_heatmap[row, col] += 1
                frame_heatmaps[frame_idx, row, col] += 1
        
        # =================================================================
        # 3. CREATE COMPREHENSIVE STATISTICS VISUALIZATION
        # =================================================================
        fig = plt.figure(figsize=(20, 16))
        
        # Main title
        fig.suptitle(f'Aggregate Token Selection Analysis - {n_steps} Steps\n'
                    f'Total selections: {total_selections} | '
                    f'Avg per step: {np.mean(tokens_per_step):.1f}',
                    fontsize=18, y=0.98)
        
        # =================================================================
        # 3.1 Create grid for subplots
        # =================================================================
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Subplot 1: Tokens per step over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(range(n_steps), tokens_per_step, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Step Index')
        ax1.set_ylabel('Tokens Selected')
        ax1.set_title('Tokens per Step Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(range(n_steps), 0, tokens_per_step, alpha=0.3)
        
        # Subplot 2: Token distribution histogram
        ax2 = fig.add_subplot(gs[0, 1])
        if all_tokens:
            ax2.hist(all_tokens, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Token Index')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Overall Token Distribution\nRange: {min(all_tokens)}-{max(all_tokens)}')
            ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Frame distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if all_frame_indices:
            frame_counts = np.bincount(all_frame_indices, minlength=self.n_frames)
            ax3.bar(range(self.n_frames), frame_counts, alpha=0.7, color='purple', edgecolor='black')
            ax3.set_xlabel('Condition Frame Index')
            ax3.set_ylabel('Token Count')
            ax3.set_title('Token Distribution Across Frames')
            ax3.set_xticks(range(self.n_frames))
            ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Row distribution
        ax4 = fig.add_subplot(gs[0, 3])
        row_sums = overall_heatmap.sum(axis=1)
        ax4.barh(range(self.grid_h), row_sums, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Token Count')
        ax4.set_ylabel('Row Index')
        ax4.set_title('Row-wise Distribution')
        ax4.set_yticks(range(0, self.grid_h, 2))
        ax4.grid(True, alpha=0.3, axis='x')
        
        # =================================================================
        # 3.2 Overall heatmap
        # =================================================================
        ax5 = fig.add_subplot(gs[1, 0:2])
        im5 = ax5.imshow(overall_heatmap, cmap='YlOrRd', aspect='equal')
        ax5.set_title(f'Overall Token Selection Heatmap\nTotal: {int(overall_heatmap.sum())} selections')
        ax5.set_xlabel('Column')
        ax5.set_ylabel('Row')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # Add grid to heatmap
        for i in range(self.grid_h + 1):
            ax5.axhline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
        for j in range(self.grid_w + 1):
            ax5.axvline(j - 0.5, color='white', alpha=0.2, linewidth=0.5)
        
        # =================================================================
        # 3.3 Column distribution
        # =================================================================
        ax6 = fig.add_subplot(gs[1, 2])
        col_sums = overall_heatmap.sum(axis=0)
        ax6.bar(range(self.grid_w), col_sums, alpha=0.7, color='red', edgecolor='black')
        ax6.set_xlabel('Column Index')
        ax6.set_ylabel('Token Count')
        ax6.set_title('Column-wise Distribution')
        ax6.set_xticks(range(0, self.grid_w, 2))
        ax6.grid(True, alpha=0.3, axis='y')
        
        # =================================================================
        # 3.4 Spatial scatter plot
        # =================================================================
        ax7 = fig.add_subplot(gs[1, 3])
        if all_rows and all_cols:
            scatter = ax7.scatter(all_cols, all_rows, 
                                 c=all_frame_indices, cmap='viridis',
                                 alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
            ax7.set_xlabel('Column Index')
            ax7.set_ylabel('Row Index')
            ax7.set_title(f'Spatial Distribution\n(Color = Frame Index)')
            ax7.set_xlim(-0.5, self.grid_w-0.5)
            ax7.set_ylim(self.grid_h-0.5, -0.5)
            ax7.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax7, label='Frame Index')
        
        # =================================================================
        # 3.5 Per-Frame Heatmaps
        # =================================================================
        frame_titles = ['Frame 0 Heatmap', 'Frame 1 Heatmap', 'Frame 2 Heatmap', 'Frame 3 Heatmap']
        for i in range(4):
            if i < self.n_frames:
                ax = fig.add_subplot(gs[2, i])
                im = ax.imshow(frame_heatmaps[i], cmap='YlOrRd', aspect='equal')
                ax.set_title(f'{frame_titles[i]}\nTotal: {int(frame_heatmaps[i].sum())}')
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # =================================================================
        # 4. SAVE THE AGGREGATE STATISTICS FIGURE
        # =================================================================
        plt.tight_layout()
        aggregate_path = os.path.join(output_dir, 'aggregate_statistics.png')
        plt.savefig(aggregate_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nAggregate statistics saved: {aggregate_path}")
        
        # =================================================================
        # 5. CREATE HIGH-RESOLUTION HEATMAP
        # =================================================================
        fig_hr = plt.figure(figsize=(10, 8))
        ax_hr = fig_hr.add_subplot(111)
        
        # Normalize heatmap for better visualization
        norm_heatmap = overall_heatmap / overall_heatmap.max() if overall_heatmap.max() > 0 else overall_heatmap
        
        im_hr = ax_hr.imshow(norm_heatmap, cmap='plasma', aspect='equal', 
                            interpolation='bilinear')
        ax_hr.set_title(f'Normalized Selection Heatmap\n{total_selections} selections across {n_steps} steps')
        ax_hr.set_xlabel('Column (16×16 grid)')
        ax_hr.set_ylabel('Row (16×16 grid)')
        
        # Add grid
        for i in range(self.grid_h + 1):
            ax_hr.axhline(i - 0.5, color='white', alpha=0.2, linewidth=0.5)
        for j in range(self.grid_w + 1):
            ax_hr.axvline(j - 0.5, color='white', alpha=0.2, linewidth=0.5)
        
        plt.colorbar(im_hr, ax=ax_hr, label='Normalized Selection Frequency')
        plt.tight_layout()
        hr_path = os.path.join(output_dir, 'highres_heatmap.png')
        plt.savefig(hr_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"High-resolution heatmap saved: {hr_path}")
        
        # =================================================================
        # 6. CREATE STATISTICAL SUMMARY TEXT FILE
        # =================================================================
        summary_path = os.path.join(output_dir, 'statistical_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("TOKEN SELECTION STATISTICAL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"ANALYSIS CONFIGURATION:\n")
            f.write(f"  Condition frames: {self.n_frames}\n")
            f.write(f"  Tokens per frame: {self.tokens_per_frame} ({self.grid_h}x{self.grid_w} grid)\n")
            f.write(f"  Total tokens: {self.total_tokens}\n")
            f.write(f"  Steps analyzed: {n_steps}\n\n")
            
            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"  Total selections: {total_selections}\n")
            f.write(f"  Average per step: {np.mean(tokens_per_step):.2f}\n")
            f.write(f"  Std deviation: {np.std(tokens_per_step):.2f}\n")
            f.write(f"  Min per step: {np.min(tokens_per_step)}\n")
            f.write(f"  Max per step: {np.max(tokens_per_step)}\n\n")
            
            f.write(f"FRAME DISTRIBUTION:\n")
            for frame_idx in range(self.n_frames):
                count = frame_heatmaps[frame_idx].sum()
                percentage = (count / total_selections * 100) if total_selections > 0 else 0
                f.write(f"  Frame {frame_idx}: {int(count)} selections ({percentage:.1f}%)\n")
            
            f.write(f"\nSPATIAL STATISTICS:\n")
            f.write(f"  Most selected row: {np.argmax(row_sums)} ({row_sums.max():.0f} selections)\n")
            f.write(f"  Most selected column: {np.argmax(col_sums)} ({col_sums.max():.0f} selections)\n")
            
            # Find top 5 most selected positions
            flat_indices = np.argsort(overall_heatmap.flatten())[::-1][:5]
            f.write(f"\nTOP 5 MOST SELECTED POSITIONS:\n")
            for i, idx in enumerate(flat_indices):
                row, col = np.unravel_index(idx, overall_heatmap.shape)
                count = overall_heatmap[row, col]
                f.write(f"  {i+1}. Row {row}, Col {col}: {count} selections\n")
        
        print(f"Statistical summary saved: {summary_path}")
        
        return {
            'overall_heatmap': overall_heatmap,
            'frame_heatmaps': frame_heatmaps,
            'tokens_per_step': tokens_per_step,
            'total_selections': total_selections,
            'n_steps': n_steps
        }

def visualize_all_steps_and_aggregate(pkl_path="1.pkl", 
                                      step_output_dir="step_visualizations",
                                      max_steps_to_visualize=10):
    """
    Visualize each step individually AND create aggregate statistics
    
    max_steps_to_visualize: Maximum number of individual steps to visualize
                            (to avoid creating too many images)
    """
    import pickle
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("=== Loading Data ===")
    print(f"Image steps: {len(data['raw_images'])}")
    print(f"Token steps: {len(data['selected_token_indices'])}")
    
    # Initialize visualizer
    visualizer = TokenVisualizer(
        n_condition_frames=4,
        tokens_per_frame=256,
        grid_size=16,
        original_image_size=(96, 96),
        model_image_size=256
    )
    
    # =================================================================
    # 1. VISUALIZE INDIVIDUAL STEPS
    # =================================================================
    print(f"\n=== Visualizing Individual Steps ===")
    
    n_steps = min(len(data["raw_images"]), len(data["selected_token_indices"]), max_steps_to_visualize)
    
    step_distributions = []
    for step_idx in range(n_steps):
        print(f"\nProcessing step {step_idx}...")
        distribution = visualizer.visualize_single_step(
            data["raw_images"][step_idx],
            data["selected_token_indices"][step_idx],
            step_idx,
            save_dir=step_output_dir
        )
        step_distributions.append(distribution)
    
    print(f"\nIndividual step visualizations saved to: {step_output_dir}")
    
    # =================================================================
    # 2. CREATE AGGREGATE STATISTICS
    # =================================================================
    print(f"\n=== Creating Aggregate Statistics ===")
    
    stats = visualizer.create_aggregate_statistics(
        data,
        output_dir=step_output_dir
    )
    
    print(f"\n=== Summary ===")
    print(f"Total selections across all {stats['n_steps']} steps: {stats['total_selections']}")
    print(f"Average tokens per step: {np.mean(stats['tokens_per_step']):.1f}")
    print(f"\nIndividual step visualizations: {step_output_dir}")
    print(f"Aggregate statistics: {step_output_dir}")
    
    return {
        'step_distributions': step_distributions,
        'aggregate_stats': stats
    }

if __name__ == "__main__":
    results = visualize_all_steps_and_aggregate(
        pkl_path="3.pkl",
        step_output_dir="output/step_visualizations",
        max_steps_to_visualize=10  # Visualize first 10 steps only
    )