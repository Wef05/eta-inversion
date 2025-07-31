# filename: skeleton_preprocess.py

import numpy as np
import torch
from skimage.morphology import skeletonize
from skimage import img_as_bool


def preprocess_skeleton(img, thr: int = 128):
    """
    对输入图片执行骨架化，自动兼容 numpy / torch 张量

    参数
    ----
    img : torch.Tensor | np.ndarray
        单通道灰度或二值图，值域 0~1
    thr : float, optional
        二值化阈值；灰度图会先按此阈值 binarize，默认为 0.5

    返回
    ----
    skel : 同 img 类型
        骨架化后二值图，像素取值 {0,1}
    """
    img_norm = (img > thr).astype(np.float32)
    # 2. 骨架化
    skel = skeletonize(img_as_bool(img_norm)).astype(np.uint8) * 255
    return skel


# =============== Demo 用法 ===============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.data import binary_blobs

    # 生成一张随机二值图做示例
    sample = binary_blobs(length=128, blob_size_fraction=0.2, volume_fraction=0.3).astype("float32")
    skel = preprocess_skeleton(sample)

    # 显示效果
    plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(sample, cmap="gray"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.title("Skeleton"); plt.imshow(skel,   cmap="gray"); plt.axis("off")
    plt.show()