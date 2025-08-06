"""
适用于基于噪声预测差异的自适应掩码合并（merge）模块  
"""

import torch
import torch.nn.functional as F
class AdaptiveMerge:
    """
    以阈值 + 方形核膨胀生成掩码，然后用该掩码将源(target[0])与目标(target[1]) 逐像素混合  
    """

    def __init__(self, lamb: float = 0.2, kernel_size: int = 3):
        """
        参数
        ----------
        lamb : float
            阈值 λ ，|diff|<=λ 的像素会被视为“相似”，进入掩码。
        kernel_size : int
            方形膨胀核尺寸 k (k×k)，必须为奇数；越大→膨胀越强。
        """
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数"
        self.lamb = lamb
        self.kernel_size = kernel_size

    # --------------------------------------------------------------------- #
    # 函数：_dilate
    # 1. 简短描述：使用方形核对二值掩码做形态学膨胀
    # 2. Input：mask[B,C,H,W]，dtype=bool / float
    # 3. Output：膨胀后的掩码，同形状 (B,C,H,W)，dtype=float32
    # 4. Method：
    #    (1) 通过 F.max_pool2d 实现膨胀（核大小=kernel_size，步长=1，padding=kernel_size//2）
    # --------------------------------------------------------------------- #
    def _dilate(self, mask: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        # 使用 max_pool2d 等价于二值膨胀
        dilated = F.max_pool2d(mask.float(), kernel_size=self.kernel_size,
                               stride=1, padding=pad)
        return dilated

    # --------------------------------------------------------------------- #
    #
    # 根据噪声差异生成并膨胀掩码
    # Method：
    #    (1) 计算逐像素绝对差 |x_t - x_s|
    #    (2) 保留多通道差异 ➜ diff_ch[1,C,H,W]
    #    (3) 阈值化 (<=λ) ➜ 初始二值掩码
    #    (4) 方形核膨胀处理小空洞
    # --------------------------------------------------------------------- #
    def get_mask(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        # 逐通道差异
        diff = (x_t - x_s).abs()             # [1,C,H,W]
        # 直接使用多通道差异，不做通道均值
        diff2d = diff  # [1,C,H,W]
        # 阈值化：相似像素 → 1，其他为 0
        mask = (diff2d <= self.lamb)
        # 膨胀
        mask = self._dilate(mask)
        return mask.clamp(0, 1)              # float32, 取值 0/1，与输入通道数一致

    def __call__(self, x_t: torch.Tensor) -> torch.Tensor:
        assert x_t.size(0) >= 2, "输入必须至少包含两帧 [src, tgt]"
        x_src, x_tgt = x_t[:1], x_t[1:2]      # 保留批次维度
        mask = self.get_mask(x_src, x_tgt)    # [1,C,H,W]
        # 广播到通道维
        x_out = x_src + mask * (x_tgt - x_src)
        return x_out


# --------------------------- DEMO / 单元测试 --------------------------- #
if __name__ == "__main__":
    B, C, H, W = 2, 4, 64, 64
    torch.manual_seed(0)
    x = torch.rand(B, C, H, W)  # 生成随机噪声预测
    merge = AdaptiveMerge(lamb=0.15, kernel_size=5)
    out = merge(x)
    print(f"输出形状: {out.shape}")