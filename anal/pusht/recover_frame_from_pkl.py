import pickle
import numpy as np
from PIL import Image
import os

# 1. 读取 pkl
with open("/home/stg/data/workspace/91MrQiao_ws/temp/unified_video_action_hot/anal/pusht/3.pkl", "rb") as f:
    data = pickle.load(f)

raw_images = data["raw_images"]
print(len(raw_images))

# 比较 step 0 frame 3 和 step 1 frame 3
a = raw_images[0][0, 1]
b = raw_images[1][0, 0]

print("allclose:", np.allclose(a, b, atol=1/255))
print("max diff:", np.abs(a - b).max())

# 2. 创建输出目录
os.makedirs("recovered_images", exist_ok=True)

# 3. 遍历 step 和 frame
for step_idx, img_tensor in enumerate(raw_images):
    # img_tensor: (1, 16, 3, 128, 128)
    img_tensor = img_tensor[0]  # (16, 3, 128, 128)

    for frame_idx in range(img_tensor.shape[0]):
        img = img_tensor[frame_idx]          # (3, 128, 128)
        img = np.transpose(img, (1, 2, 0))   # (128, 128, 3)
        img = (img * 255).clip(0, 255).astype(np.uint8)

        Image.fromarray(img).save(
            f"recovered_images/step_{step_idx:02d}_frame_{frame_idx:02d}.png"
        )
print("saving")
