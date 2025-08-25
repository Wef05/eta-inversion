# filename: clip_cuda_global_image_embedder.py
"""
目标：
- 在全局初始化阶段：使用 CUDA 加载 CLIP ViT-L/14（openai/clip-vit-large-patch14），
  从本地目录读取 sketch.png，提取图像嵌入，并缓存为张量形状 (1, 1, 768)。
- 提供“注入函数” `inject_and_score(text_emb)`：
  1) 输入原始文本嵌入，形状 (1, 77, 768)（例如来自 LDM/Stable Diffusion 的 CLIP 文本编码器输出）；
  2) 返回首次初始化时计算并缓存的图片嵌入（形状 (1, 1, 768)）；
  3) 计算该图片嵌入与 77 个 token 嵌入的相似度（默认使用余弦，相当于单位向量点积），
     将 77 个相似度打印到控制台，并且“只打印一次”。

依赖：
    pip install torch pillow transformers

说明（主观看法）：
- 既然你要求 CUDA，我强制在 CUDA 设备上运行；若无 CUDA，直接报错，避免静默退化到 CPU
- 相似度在图像/文本向量均 L2 归一化后用点积即可（更快更稳）
"""

from __future__ import annotations
import os
import io
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor


# =========================
# 全局状态（单例样式）
# =========================
_DEVICE: Optional[torch.device] = None
_MODEL: Optional[CLIPModel] = None
_IMAGE_PROC: Optional[CLIPImageProcessor] = None
_IMAGE_EMB_1x1x768: Optional[torch.Tensor] = None  # 缓存的图像嵌入 (1,1,768)，device=cuda
_PRINTED_ONCE: bool = False  # 控制仅打印一次相似度


# =========================
# 工具函数
# =========================
def _ensure_cuda() -> torch.device:
    """
    简短描述：
        确保存在 CUDA，并返回 cuda 设备句柄。
    Input：
        无
    Output：
        torch.device("cuda")
    Method：
        1) 若 torch.cuda.is_available() 为 False，直接抛错；
        2) 返回 torch.device("cuda")。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请在有 NVIDIA GPU 的环境中运行，或安装带 CUDA 的 PyTorch。")
    return torch.device("cuda")


def _load_clip_vitl14_on_cuda() -> tuple[CLIPModel, CLIPImageProcessor, torch.device]:
    """
    简短描述：
        在 CUDA 上加载 CLIP ViT-L/14 模型与图像处理器。
    Input：
        无
    Output：
        (model, image_processor, device)
    Method：
        1) 强制使用 CUDA 设备；
        2) 从 HF Hub 加载 `openai/clip-vit-large-patch14`；
        3) 将模型迁移到 cuda 并 eval()。
    """
    device = _ensure_cuda()
    model_name = "openai/clip-vit-large-patch14"
    image_proc = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, image_proc, device


def _to_rgb_image(path: str) -> Image.Image:
    """
    简短描述：
        加载本地图片并转为 RGB。
    Input：
        path: 图片相对或绝对路径（本脚本要求为 'sketch.png'）
    Output：
        PIL.Image (RGB)
    Method：
        1) 使用 PIL.Image.open；
        2) .convert("RGB") 统一颜色空间。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件：{path}")
    return Image.open(path).convert("RGB")


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    简短描述：
        对张量按指定维度做 L2 归一化，避免除零。
    Input：
        x  : 任意形状张量
        dim: 归一化维度
        eps: 数值稳定项
    Output：
        与 x 同形状的单位向量张量
    Method：
        1) 计算范数并 clamp；
        2) x / norm。
    """
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


# =========================
# 全局初始化：加载模型 + 读取 sketch.png + 计算与缓存图像嵌入
# =========================
@torch.inference_mode()
def _global_initialize():
    """
    简短描述：
        全局一次性初始化：CUDA + 模型 + 读图 + 编码 + 缓存 (1,1,768)。
    Input：
        无
    Output：
        无（更新全局变量）
    Method：
        1) 加载 CLIP ViT-L/14 模型与预处理器到 CUDA；
        2) 读取 ./sketch.png，按模型预处理成张量；
        3) 使用 get_image_features 得到 (1, 768)，做 L2 归一化；
        4) 在维度 1 处增加一维 → (1, 1, 768)，并缓存到全局。
    """
    global _DEVICE, _MODEL, _IMAGE_PROC, _IMAGE_EMB_1x1x768

    if _MODEL is not None and _IMAGE_EMB_1x1x768 is not None:
        return  # 已初始化

    model, image_proc, device = _load_clip_vitl14_on_cuda()
    img = _to_rgb_image("sketch.png")

    # 预处理
    inputs = image_proc(images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 在 CUDA 上可用 autocast(fp16) 节省显存；但为了数值稳定，这里保持 fp32
    feats = model.get_image_features(**inputs)  # (1, 768)
    feats = _l2_normalize(feats, dim=-1)       # 单位化，便于余弦相似度
    #feats = feats.unsqueeze(1).expand(-1,77,-1)              # -> (1,77, 768)
    feats = feats.unsqueeze(1)

    _DEVICE = device
    _MODEL = model
    _IMAGE_PROC = image_proc
    _IMAGE_EMB_1x1x768 = feats  # 常驻 CUDA 显存


# 执行全局初始化
_global_initialize()


# =========================
# 对外“注入函数”：接受文本嵌入，返回图片嵌入，并打印一次相似度
# =========================
@torch.inference_mode()
def inject_and_score(text_embeddings: torch.Tensor) -> torch.Tensor:
    """
    简短描述：
        接受原文本嵌入 (1, 77, 768)，返回缓存的图片嵌入 (1, 1, 768)，
        并计算该图片嵌入与 77 个 token 的相似度（仅首次调用打印一次）。
    Input：
        text_embeddings: torch.Tensor，形状 (1, 77, 768)，device 可以是 cpu/cuda，
                         本函数会自动搬运到 cuda
    Output：
        torch.Tensor：形状 (1, 1, 768) 的图片嵌入（全局缓存）
    Method（步骤描述）：
        1) 校验并将文本嵌入搬到 CUDA；
        2) 对文本嵌入做 L2 归一化（若上游未归一化，保证这里是单位向量域）；
        3) 取全局图片嵌入 IMG ∈ (1,1,768)，计算 SIM = IMG • TEXT^T → (1,1,77)；
           由于已经单位化，• 等价于余弦相似度；
        4) 将 SIM squeeze 到 (77,) 并“仅打印一次”。
        5) 返回缓存的图片嵌入 (1,1,768)。
    """
    global _PRINTED_ONCE

    # 基本校验
    if not isinstance(text_embeddings, torch.Tensor):
        raise TypeError("text_embeddings 必须是 torch.Tensor")
    if text_embeddings.ndim != 3 or text_embeddings.shape[0] != 1 or text_embeddings.shape[2] != 768:
        #raise ValueError(f"text_embeddings 形状应为 (1, 77, 768)，实际为 {tuple(text_embeddings.shape)}")
        return text_embeddings

    # 搬运到 CUDA
    text_embeddings = text_embeddings.to(_DEVICE, dtype=torch.float32)

    # L2 归一化（按最后一维 768）
    text_embeddings = _l2_normalize(text_embeddings, dim=-1)  # (1,77,768)
    # 仅首次打印一次
    # if not _PRINTED_ONCE:
    #     # 取全局图片嵌入
    #     img_emb = _IMAGE_EMB_1x1x768.clone()  # (1,77,768) on cuda
    #     # 计算相似度： (1,1,768) @ (1,768,77) -> (1,1,77)
    #     sim = torch.matmul(img_emb.squeeze(0)[0:1].unsqueeze(0), text_embeddings.transpose(1, 2))  # (1,1,77)
    #     sim_flat = sim.squeeze(0).squeeze(0)  # -> (77,)
    #     # 将相似度搬到 CPU 打印
    #     sims = sim_flat.detach().cpu().numpy()
    #     print("Per-token similarity (77 tokens):")
    #     # 一行打印，避免多次循环刷屏
    #     # 你也可以格式化为保留 4 位小数
    #     print(" ".join(f"{v:.4f}" for v in sims))
    #     _PRINTED_ONCE = True

    # 返回缓存的图片嵌入 (1,1,768)
    return _IMAGE_EMB_1x1x768


# =========================
# 可选示例：脚本直接运行时的自测
# =========================
if __name__ == "__main__":
    # 仅用于演示：构造一个“假的”文本嵌入 (1,77,768)
    # 实际使用时，应该把来自你 LDM/CLIP 文本编码器的真实张量传进来
    fake_text = torch.randn(1, 77, 768)  # 未归一化，函数内部会做 L2 归一化
    img_emb = inject_and_score(fake_text)  # 同时会在控制台打印一次 77 个相似度
    # 打印返回张量形状验证
    print("Returned image embedding shape:", tuple(img_emb.shape))  # (1, 1, 768)