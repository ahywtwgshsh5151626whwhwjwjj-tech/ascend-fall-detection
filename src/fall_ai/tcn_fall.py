#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tcn_fall.py — W3 时序分类头: 模型定义 + numpy 前向推理

设计要点
--------
* 输入  (B, C, T)  C=54 (17*3 关键点 + 3 派生量), T=30 (=1s@30fps)
* 结构  膨胀因果卷积 TCN, 感受野覆盖整段 1 秒
* 参数  ≈10 万 (实测 100,290), 对板端 CPU 而言仍然极小
* 部署  板端用 numpy 手写前向, 不占 NPU 预算(NPU 单帧已 41ms)

numpy 前向与 PyTorch 前向必须数值一致 —— 见 selftest 的逐元素比对。
"""
from __future__ import annotations

import numpy as np

N_POSE = 51          # 17 * 3
N_EXTRA = 3          # bbox横比, 模型分数, 关键点均分
N_CH = N_POSE + N_EXTRA
WIN = 30
N_CLASS = 2
CLASS_NAMES = ["normal", "fall"]


# ================================================================ PyTorch
def build_torch_model(n_ch=N_CH, width=64, n_class=N_CLASS):
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, c, dilation):
            super().__init__()
            self.conv = nn.Conv1d(c, c, kernel_size=5, dilation=dilation,
                                  padding=2 * dilation)
            self.bn = nn.BatchNorm1d(c)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(x + self.bn(self.conv(x)))

    class TCNFall(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(
                nn.Conv1d(n_ch, width, kernel_size=5, padding=2),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                ResBlock(width, 1), ResBlock(width, 2),
                ResBlock(width, 4), ResBlock(width, 8),
            )
            self.head = nn.Linear(width, n_class)

        def forward(self, x):                 # (B, C, T)
            h = self.blocks(self.inp(x))
            h = h.mean(dim=2)                 # 全局平均池化
            return self.head(h)

        def prob(self, x):
            return self.forward(x).softmax(dim=-1)

    return TCNFall()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ================================================================ numpy 前向
def _conv1d(x, w, b, pad=0, dilation=1):
    """x:(B,Cin,T) w:(Cout,Cin,K) b:(Cout,)"""
    B, Cin, T = x.shape
    Cout, _, K = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    T_out = x.shape[2] - (K - 1) * dilation
    out = np.zeros((B, Cout, T_out), np.float32)
    for k in range(K):
        seg = x[:, :, k * dilation: k * dilation + T_out]
        out += np.einsum("bct,oc->bot", seg, w[:, :, k], optimize=True)
    return out + b.reshape(1, -1, 1)


def _bn_eval(x, gamma, beta, mean, var, eps=1e-5):
    scale = (gamma / np.sqrt(var + eps)).astype(np.float32).reshape(1, -1, 1)
    shift = (beta - gamma * mean / np.sqrt(var + eps)).astype(np.float32).reshape(1, -1, 1)
    return x * scale + shift


class NumpyTCN:
    """从 PyTorch state_dict 抽取权重, 纯 numpy 前向。"""

    def __init__(self, weights: dict):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    @classmethod
    def from_torch(cls, model):
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        return cls(sd)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls({k: z[k] for k in z.files})

    def save(self, path):
        np.savez_compressed(path, **self.w)

    def _bn(self, x, prefix):
        w = self.w
        return _bn_eval(x, w[prefix + ".weight"], w[prefix + ".bias"],
                        w[prefix + ".running_mean"], w[prefix + ".running_var"])

    def forward(self, x):
        """x: (B, C, T) -> logits (B, 2)"""
        w = self.w
        h = _conv1d(x, w["inp.0.weight"], w["inp.0.bias"], pad=2)
        h = np.maximum(self._bn(h, "inp.1"), 0.0)

        for i in range(4):
            p = f"blocks.{i}"
            dil = [1, 2, 4, 8][i]
            r = _conv1d(h, w[p + ".conv.weight"], w[p + ".conv.bias"],
                        pad=2 * dil, dilation=dil)
            r = self._bn(r, p + ".bn")
            h = np.maximum(h + r, 0.0)

        h = h.mean(axis=2)
        return h @ w["head.weight"].T + w["head.bias"]

    def prob(self, x):
        z = self.forward(x)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# ================================================================ 特征组装
def build_input(kpts, extra, valid, include_hip_h=False):
    """kpts:(T,17,3) extra:(T,4) valid:(T,) -> (C,T) float32

    include_hip_h=False 时丢弃 extra[:,0](帧内髋高), 因为它是机位相关的,
    跨数据集/跨机位会拖累泛化。其余 3 个派生量保留。
    """
    T = kpts.shape[0]
    pose = kpts.reshape(T, N_POSE).T                      # (51,T)
    cols = [1, 2, 3] if not include_hip_h else [0, 1, 2, 3]
    ex = extra[:, cols].T                                  # (E,T)
    x = np.concatenate([pose, ex], axis=0).astype(np.float32)
    x[:, valid < 0.5] = 0.0
    return x


def input_dim(include_hip_h=False):
    return N_POSE + (4 if include_hip_h else N_EXTRA)
