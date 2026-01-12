import zarr
import numpy as np

SRC = "data/pusht/pusht_cchi_v7_replay.zarr"
DST = "data/pusht/pusht_one_episode.zarr"
EP_ID = 0

src = zarr.open(SRC, mode="r")
dst = zarr.open(DST, mode="w")

episode_ends = src["meta/episode_ends"][:]

start = 0 if EP_ID == 0 else episode_ends[EP_ID - 1] + 1
end = episode_ends[EP_ID]

print(f"Episode {EP_ID}: frames {start} ~ {end}")

# ---------- data ----------
for key in ["img", "action", "state", "keypoint", "n_contacts"]:
    src_key = f"data/{key}"
    if src_key not in src:
        continue

    data = np.asarray(src[src_key][start:end+1])

    arr = dst.create_array(
        src_key,
        shape=data.shape,
        dtype=data.dtype,
        overwrite=True,
    )
    arr[:] = data

# ---------- meta ----------
length = end - start + 1
meta = dst.create_array(
    "meta/episode_ends",
    shape=(1,),
    dtype=np.int64,
    overwrite=True,
)
meta[0] = length - 1

print("Saved to:", DST)

