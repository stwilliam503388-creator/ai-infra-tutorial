"""Triton vector add 示例——第2章 算子层配套代码。

运行: python vector_add.py  (需要 GPU 且已 pip install -r requirements.txt)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ① 并行: 每个 program 负责一个 BLOCK_SIZE 大小的分块
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # ② 掩码: 数组末尾不足一块时防止越界
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # ③ 分块: 算出需要多少个块
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1_000_000, device="cuda")
    y = torch.randn(1_000_000, device="cuda")
    out = vector_add(x, y)
    assert torch.allclose(out, x + y), "Triton 结果与 torch 不一致"
    print("OK: Triton vector add 结果与 torch 一致")
