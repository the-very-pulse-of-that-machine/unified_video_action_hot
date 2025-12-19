import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
import time
from collections import OrderedDict
from omegaconf import open_dict
import pickle

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner


# ===============================
#     Timing System (Layer + Module)
# ===============================

LAYER_TIME = OrderedDict()
MODULE_TIME = OrderedDict()


def get_two_level_module(name: str):
    """
    二级结构聚类 module name，如：
      model.encoder.xxx → model.encoder
      vae_model.decoder.xxx → vae_model.decoder
      conv_stem → conv_stem
    """
    parts = name.split(".")
    if len(parts) >= 2:
        return parts[0] + "." + parts[1]   # 二级结构
    return parts[0]                         # 只有一级结构


def register_time_hooks(model):
    use_cuda = next(model.parameters()).is_cuda

    for name, module in model.named_modules():
        if name == "":
            continue

        layer_name = name
        module_name = get_two_level_module(name)

        def _make_hook(layer_name, module_name):
            def pre_hook(m, inp):
                if use_cuda:
                    m.__start_event = torch.cuda.Event(enable_timing=True)
                    m.__end_event = torch.cuda.Event(enable_timing=True)
                    m.__start_event.record()
                else:
                    m.__start_time = time.time()

            def fwd_hook(m, inp, out):
                if use_cuda:
                    m.__end_event.record()
                    torch.cuda.synchronize()
                    elapsed = m.__start_event.elapsed_time(m.__end_event)  # ms
                else:
                    elapsed = (time.time() - m.__start_time) * 1000

                LAYER_TIME.setdefault(layer_name, []).append(float(elapsed))
                MODULE_TIME.setdefault(module_name, []).append(float(elapsed))

            return pre_hook, fwd_hook

        pre_hook, fwd_hook = _make_hook(layer_name, module_name)
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(fwd_hook)



# ===============================
#               MAIN
# ===============================

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--dataset_path", required=False)
@click.option("--n_test", default=1, help="Number of test episodes to run")
@click.option("--n_test_vis", default=1, help="Number of test episodes to visualize")
@click.option("--n_train", default=0, help="Number of train episodes to run")
@click.option("--n_train_vis", default=0, help="Number of train episodes to visualize")
@click.option("--save_data", default=True, help="Save image and token data")
@click.option("--num_frames", default=4, help="Number of frames to process (1 or 4)")
@click.option("--max_save_steps", default=10, help="Maximum number of steps to save data for")
def main(checkpoint, output_dir, device, dataset_path, n_test, n_test_vis, n_train, n_train_vis, save_data, num_frames, max_save_steps):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        if dataset_path is not None:
            cfg.task.dataset.dataset_path = dataset_path
        
        # Override env_runner parameters for single episode
        cfg.task.env_runner.n_test = n_test
        cfg.task.env_runner.n_test_vis = n_test_vis
        cfg.task.env_runner.n_train = n_train
        cfg.task.env_runner.n_train_vis = n_train_vis

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload)

    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # 注册二级结构 timing hooks
    print("[Timing] Registering hierarchical hooks (2-level)...")
    register_time_hooks(policy)

    # Run environments
    env_runners = load_env_runner(cfg, output_dir)
    
    # 用于保存数据的列表
    saved_data = {
        'image_data': [],
        'selected_token_indices': [],
        'actions': [],
        'timesteps': []
    }

    if "libero" in cfg.task.name:
        step_log = {}
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)

        all_test_mean_score = {
            k: v for k, v in step_log.items()
            if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))
        runner_log = step_log

    else:
        env_runner = env_runners
        
        # 修改：保存图像数据和token数据
        if save_data:
            # 获取模型
            model = policy.model
            
            # 用于保存数据的列表
            saved_data = {
                'sample_tokens_cond': [],  # 只保存sample_tokens中的cond数据
                'selected_token_indices': [],  # 保存select token index
                'step_count': 0  # 记录当前保存的step数
            }
            
            # 添加hook来捕获sample_tokens方法中的cond数据
            if hasattr(model, 'sample_tokens'):
                original_sample_tokens = model.sample_tokens
                
                def wrapped_sample_tokens(bsz, cond, text_latents=None, num_iter=64, cfg=1.0, 
                                         cfg_schedule="linear", temperature=1.0, progress=False,
                                         history_nactions=None, nactions=None, proprioception_input={},
                                         task_mode=None, vae_model=None, x=None):
                    # 保存sample_tokens中的cond数据
                    if saved_data['step_count'] < max_save_steps:
                        cond_np = cond.detach().cpu().numpy()
                        saved_data['sample_tokens_cond'].append(cond_np)
                        saved_data['step_count'] += 1
                    
                    # 调用原始sample_tokens方法
                    return original_sample_tokens(bsz, cond, text_latents, num_iter, cfg, cfg_schedule,
                                                 temperature, progress, history_nactions, nactions,
                                                 proprioception_input, task_mode, vae_model, x)
                
                # 临时替换sample_tokens方法
                model.sample_tokens = wrapped_sample_tokens
            
            # 添加hook来捕获select token index
            original_forward_mae_encoder = model.forward_mae_encoder
            
            def wrapped_forward_mae_encoder(x, mask, cond, text_latents=None, history_nactions=None, 
                                           nactions=None, task_mode=None, proprioception_input={}):
                result = original_forward_mae_encoder(x, mask, cond, text_latents, history_nactions, 
                                                     nactions, task_mode, proprioception_input)
                # 保存select token index
                if hasattr(model, 'selected_token_index') and model.selected_token_index is not None:
                    if len(saved_data['selected_token_indices']) < max_save_steps:
                        saved_data['selected_token_indices'].append(model.selected_token_index.detach().cpu().numpy())
                return result
            
            # 临时替换forward_mae_encoder方法
            model.forward_mae_encoder = wrapped_forward_mae_encoder
            
            # 运行评估
            runner_log = env_runner.run(policy)
            
            # 恢复原始方法
            model.forward_mae_encoder = original_forward_mae_encoder
            if hasattr(model, 'sample_tokens'):
                model.sample_tokens = original_sample_tokens
            
            # 限制保存的数据量
            if len(saved_data['sample_tokens_cond']) > max_save_steps:
                saved_data['sample_tokens_cond'] = saved_data['sample_tokens_cond'][:max_save_steps]
            if len(saved_data['selected_token_indices']) > max_save_steps:
                saved_data['selected_token_indices'] = saved_data['selected_token_indices'][:max_save_steps]
            
            # 保存数据到文件
            data_path = os.path.join(output_dir, "saved_data.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(saved_data, f)
            print(f"[Data] Saved data → {data_path}")
            
            # 打印数据统计
            print(f"[Data] Collected {len(saved_data['sample_tokens_cond'])} sample_tokens cond data")
            if saved_data['sample_tokens_cond']:
                first_cond = saved_data['sample_tokens_cond'][0]
                print(f"[Data] First sample_tokens cond shape: {first_cond.shape}")
            
            if saved_data['selected_token_indices']:
                print(f"[Data] Collected {len(saved_data['selected_token_indices'])} select token index sets")
                # 打印第一个batch的select token index
                first_indices = saved_data['selected_token_indices'][0]
                print(f"[Data] First batch select token indices shape: {first_indices.shape}")
                print(f"[Data] First batch select token indices (first 10): {first_indices[0][:10] if len(first_indices.shape) > 1 else first_indices[:10]}")
        else:
            runner_log = env_runner.run(policy)

    # =============================
    # Save timing results
    # =============================
    def agg(d):
        return {
            "sum_ms":  {k: float(np.sum(v)) for k, v in d.items()},
            "mean_ms": {k: float(np.mean(v)) for k, v in d.items()},
            "max_ms":  {k: float(np.max(v)) for k, v in d.items()},
        }

    timing_result = {
        "layer": agg(LAYER_TIME),
        "module": agg(MODULE_TIME)
    }

    timing_path = os.path.join(output_dir, "timing_stats.json")
    with open(timing_path, "w") as f:
        json.dump(timing_result, f, indent=2, sort_keys=True)

    print(f"[Timing] Saved fine-grained stats → {timing_path}")

    # =============================
    # Save evaluation log
    # =============================
    json_log = {}
    for k, v in runner_log.items():
        if isinstance(v, wandb.sdk.data_types.video.Video):
            json_log[k] = v._path
        else:
            json_log[k] = v

    out_path = os.path.join(output_dir, f"eval_log_{os.path.basename(checkpoint)}.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"[Eval] Saved → {out_path}")


if __name__ == "__main__":
    main()
