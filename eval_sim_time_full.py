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
def main(checkpoint, output_dir, device):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir

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

