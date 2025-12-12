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

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner


# ==============================
#   Timing Hooks
# ==============================

MODULE_TIME = OrderedDict()


def register_time_hooks(model):
    """
    Register forward_pre_hook & forward_hook for every module to record elapsed time.
    """
    for name, module in model.named_modules():

        if name == "":
            continue  # skip top module

        def _make_hook(m_name):
            def pre_hook(module, input):
                module.__start_time = time.time()

            def fwd_hook(module, input, output):
                elapsed = (time.time() - module.__start_time) * 1000
                MODULE_TIME.setdefault(m_name, []).append(elapsed)

            return pre_hook, fwd_hook

        pre_hook, fwd_hook = _make_hook(name)
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(fwd_hook)


# ==============================
#       Main Program
# ==============================

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, device):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # Set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir

    # Configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # -----------------------------
    # Get policy
    # -----------------------------
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # -----------------------------
    # Register timing hooks
    # -----------------------------
    print("[Timing] Registering per-layer forward hooks...")
    register_time_hooks(policy)

    # -----------------------------
    # Load environment & run
    # -----------------------------
    env_runners = load_env_runner(cfg, output_dir)

    if "libero" in cfg.task.name:
        step_log = {}
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)
            print(step_log)

        assert "test_mean_score" not in step_log
        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))
        runner_log = step_log

    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)

    # -----------------------------
    # Save timing results
    # -----------------------------
    print("[Timing] Collecting per-layer timing results...")

    layer_time_sum = {k: float(np.sum(v)) for k, v in MODULE_TIME.items()}
    layer_time_mean = {k: float(np.mean(v)) for k, v in MODULE_TIME.items()}
    layer_time_max = {k: float(np.max(v)) for k, v in MODULE_TIME.items()}

    timing_path = os.path.join(output_dir, "model_layer_time.json")
    with open(timing_path, "w") as f:
        json.dump(
            {
                "sum_ms": layer_time_sum,
                "mean_ms": layer_time_mean,
                "max_ms": layer_time_max,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"[Timing] Saved per-layer timing to: {timing_path}")

    # -----------------------------
    # Dump evaluation log
    # -----------------------------
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    for k, v in json_log.items():
        print(k, v)

    out_path = os.path.join(output_dir, f'eval_log_{checkpoint.split("/")[-1]}.json')
    print("Saving log to %s" % out_path)
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

