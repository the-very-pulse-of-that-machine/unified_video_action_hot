import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import random
import numpy as np
import pickle
from omegaconf import open_dict

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--dataset_path", required=False)
@click.option("--n_test", default=1)
@click.option("--max_save_steps", default=10)
def main(checkpoint, output_dir, device, dataset_path, n_test, max_save_steps):

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
        cfg.task.env_runner.n_test = n_test
        cfg.task.env_runner.n_test_vis = 0
        cfg.task.env_runner.n_train = 0
        cfg.task.env_runner.n_train_vis = 0

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload)

    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    env_runners = load_env_runner(cfg, output_dir)
    env_runner = env_runners

    model = policy.model

    # ===============================
    # Data buffer
    # ===============================
    saved_data = {
        "images": [],
        "selected_token_indices": [],
        "patch_size": None,
        "image_shape": None,
        "step_count": 0,
    }

    # patch size
    if hasattr(model, "patch_size"):
        saved_data["patch_size"] = model.patch_size
    elif hasattr(model, "vae_model") and hasattr(model.vae_model, "patch_size"):
        saved_data["patch_size"] = model.vae_model.patch_size

    # ===============================
    # Hook MAE encoder
    # ===============================
    original_forward_mae_encoder = model.forward_mae_encoder

    def wrapped_forward_mae_encoder(
        x, mask, cond,
        text_latents=None,
        history_nactions=None,
        nactions=None,
        task_mode=None,
        proprioception_input={}
    ):
        if saved_data["step_count"] < max_save_steps:
            if saved_data["image_shape"] is None:
                saved_data["image_shape"] = tuple(x.shape)

            saved_data["images"].append(
                x.detach().cpu().numpy()
            )

        out = original_forward_mae_encoder(
            x, mask, cond,
            text_latents,
            history_nactions,
            nactions,
            task_mode,
            proprioception_input
        )

        if (
            hasattr(model, "selected_token_index")
            and model.selected_token_index is not None
            and len(saved_data["selected_token_indices"]) < max_save_steps
        ):
            saved_data["selected_token_indices"].append(
                model.selected_token_index.detach().cpu().numpy()
            )
            saved_data["step_count"] += 1

        return out

    model.forward_mae_encoder = wrapped_forward_mae_encoder

    # ===============================
    # Run one episode
    # ===============================
    print("[Run] Collecting data...")
    env_runner.run(policy)

    # restore
    model.forward_mae_encoder = original_forward_mae_encoder

    # ===============================
    # Save
    # ===============================
    out_path = os.path.join(output_dir, "saved_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(saved_data, f)

    print(f"[Saved] → {out_path}")
    print(f"Steps saved: {saved_data['step_count']}")
    print(f"Patch size: {saved_data['patch_size']}")
    print(f"Image shape: {saved_data['image_shape']}")


if __name__ == "__main__":
    main()
