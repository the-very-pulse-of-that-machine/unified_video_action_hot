import os
import zarr
import imageio
import numpy as np
from tqdm import tqdm

ZARR_PATH = "data/pusht/pusht_one_episode.zarr"
OUT_DIR = "data/pusht/pusht_frames_one_episode"

def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    img: (H, W, 3), float32 or uint8
    """
    if img.dtype == np.uint8:
        return img

    # 常见情况：float32 in [0, 1]
    if img.max() <= 1.0:
        img = img * 255.0

    return img.clip(0, 255).astype(np.uint8)

def main():
    root = zarr.open(ZARR_PATH, mode="r")

    imgs = root["data/img"]                 # (T, H, W, 3)
    episode_ends = root["meta/episode_ends"][:]

    os.makedirs(OUT_DIR, exist_ok=True)

    start = 0
    for ep_id, end in enumerate(episode_ends):
        ep_dir = os.path.join(OUT_DIR, f"episode_{ep_id:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        print(f"Saving episode {ep_id}, frames {start} ~ {end}")

        for t in tqdm(range(start, end + 1)):
            frame = np.asarray(imgs[t])     # zarr → numpy
            frame = to_uint8(frame)

            frame_path = os.path.join(ep_dir, f"{t-start:06d}.png")
            imageio.imwrite(frame_path, frame)

        start = end + 1

    print("Done.")

if __name__ == "__main__":
    main()

