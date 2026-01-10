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
from omegaconf import open_dict

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner

# ============================================================
# CUDA Timer (GPU-only, non-intrusive)
# ============================================================

class CUDATimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()

    def elapsed(self):
        # must call torch.cuda.synchronize() outside
        return float(self.start.elapsed_time(self.end))  # ms


# ============================================================
# 🔥 BIG MODULE FORWARD TIMING (核心修改点)
# ============================================================

def forward_with_big_module_timing(policy, obs_dict, image_key):
    """
    Run ONE forward pass and record GPU time of big modules.
    Modify ONLY this function according to your model structure.
    """
    timings = {}

    with torch.no_grad():

        # ====================================================
        # 1. Vision Encoder
        # ====================================================
        with CUDATimer() as t:
            vision_feat = policy.vision_encoder(obs_dict[image_key])
        timings["vision_encoder"] = t

        # ====================================================
        # 2. Language Encoder (optional)
        # ====================================================
        if hasattr(policy, "language_encoder"):
            with CUDATimer() as t:
                lang_feat = policy.language_encoder(
                    ["dummy"] * obs_dict[image_key].shape[0]
                )
            timings["language_encoder"] = t
        else:
            lang_feat = None

        # ====================================================
        # 3. Policy Core / Transformer
        # ====================================================
        with CUDATimer() as t:
            fused_feat = policy.policy_core(vision_feat, lang_feat)
        timings["policy_core"] = t

        # ====================================================
        # 4. Action Head / Decoder
        # ====================================================
        with CUDATimer() as t:
            action = policy.action_head(fused_feat)
        timings["action_head"] = t

    return timings


# ============================================================
# Run one forward & print timing
# ============================================================

def profile_forward(policy, example_obs, device, image_key):
    policy.eval()

    obs_dict = {
        k: torch.from_numpy(v).to(device)
        for k, v in example_obs.items()
    }

    timings = forward_with_big_module_timing(
        policy, obs_dict, image_key
    )

    # ⭐ 同步一次，避免每个模块单独 sync
    torch.cuda.synchronize()

    report = {k: v.elapsed() for k, v in timings.items()}

    print("\n========== Big Module GPU Time (ms) ==========")
    for k, v in report.items():
        print(f"{k:25s}: {v:8.3f} ms")
    print("=============================================\n")

    return report


# ============================================================
# MAIN
# ============================================================

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--dataset_path", required=False)
def main(checkpoint, output_dir, device, dataset_path):

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

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload)

    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # ========================================================
    # Run env once to get example observation
    # ========================================================

    env_runners = load_env_runner(cfg, output_dir)
    example_env = (
        env_runners[0] if isinstance(env_runners, list) else env_runners
    )
    example_obs = example_env.env.reset()

    print("[Profiler] Running single forward timing ...")

    image_key = (
        "agentview_image"
        if "libero" in cfg.task.name
        else "image"
    )

    timing_report = profile_forward(
        policy, example_obs, device, image_key
    )

    # ========================================================
    # Save timing result
    # ========================================================

    out_path = os.path.join(output_dir, "big_module_timing.json")
    with open(out_path, "w") as f:
        json.dump(timing_report, f, indent=2, sort_keys=True)

    print(f"[Timing] Saved → {out_path}")

    # ========================================================
    # Normal evaluation (unchanged)
    # ========================================================

    if isinstance(env_runners, list):
        step_log = {}
        for env_runner in env_runners:
            step_log.update(env_runner.run(policy))
        runner_log = step_log
    else:
        runner_log = env_runners.run(policy)

    json_log = {}
    for k, v in runner_log.items():
        if isinstance(v, wandb.sdk.data_types.video.Video):
            json_log[k] = v._path
        else:
            json_log[k] = v

    eval_out = os.path.join(
        output_dir,
        f"eval_log_{os.path.basename(checkpoint)}.json"
    )
    json.dump(json_log, open(eval_out, "w"), indent=2)
    print(f"[Eval] Saved → {eval_out}")


if __name__ == "__main__":
    main()
