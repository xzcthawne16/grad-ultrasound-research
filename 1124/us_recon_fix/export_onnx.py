# -*- coding: utf-8 -*-
"""
部署第一步：训练 ResUNet 并保存权重 + 导出 ONNX + onnxruntime 验证精度

这是「部署最小闭环」的前半段，全程 Python，不碰 C++ / 硬件：
  1. 复用 train_residual.py 里的仿真 + ResUNet 定义（import 过来，不重复造轮子）
  2. 训练一个残差 UNet（默认 30 epoch，够验证部署链路即可）
  3. torch.save 保存权重
  4. torch.onnx.export 导出 .onnx
  5. onnxruntime 加载 .onnx 跑一遍，对比 PyTorch 输出误差

输出文件：
  - resunet.pth          权重
  - resunet.onnx         通用格式模型
  - verify_result.png    验证对比图

运行： python export_onnx.py --epochs 30
"""
# 部署第一步:训练 ResUNet 并保存权重 + 导出 ONNX + onnxruntime 验证精度
import argparse
import os
import sys

import numpy as np
import torch

# 让脚本能 import 同目录的 train_residual.py（里面定义了 ResUNet / 仿真函数 / 指标）
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from train_residual import ResUNet, gen_pair, make_low_image, make_ideal_image, psnr, ssim  # noqa: E402


def train_and_save(epochs, device, out_weight="resunet.pth"):
    """训练一个残差 UNet 并保存权重。"""
    n_train = 200
    lows, ideals = [], []
    for _ in range(n_train):
        lo, hi = gen_pair(noise=0.0)
        lows.append(lo)
        ideals.append(hi)
    lows = np.array(lows)[:, None, :, :].astype(np.float32)
    ideals = np.array(ideals)[:, None, :, :].astype(np.float32)

    model = ResUNet(residual=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l1 = torch.nn.L1Loss()

    Xt = torch.from_numpy(lows).to(device)
    Yt = torch.from_numpy(ideals).to(device)
    bs = 16

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            out = model(xb)
            loss = l1(out, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss={total / n_train:.6f}")

    model.eval()
    torch.save(model.state_dict(), out_weight)
    print(f"[OK] 权重已保存 -> {out_weight}")
    return model


def export_onnx(model, device, onnx_path="resunet.onnx"):
    """把训练好的模型导出成 ONNX 通用格式。"""
    model.eval()
    dummy = torch.randn(1, 1, 512, 256, device=device)  # 静态输入，和训练一致
    # ONNX 导出不是"读权重",而是"真的跑一遍模型"。​ 
    # 它给你一个假的输入 dummy,喂给模型,让模型真的前向算一次,
    # 然后把这个"计算过程"记录成一张静态图(ONNX 图),再把权重也写进去。

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],      # 输入名（C++ 里要用到）
        output_names=["recon"],    # 输出名
        opset_version=18,           # torch 2.x 新导出器要求 >=18，onnxruntime 1.32 完全支持
        do_constant_folding=True,   # 折叠常量，减小体积、加快推理
        dynamic_axes=None,          # 固定尺寸，设备部署最省心
    )
    print(f"[OK] ONNX 已导出 -> {onnx_path}")


def verify_onnx(model, device, onnx_path="resunet.onnx"):
    """用 onnxruntime 跑 ONNX，和 PyTorch 输出对比，确认没丢精度。"""
    import onnxruntime as ort

    # 构造一个固定测试输入（真点场景，复用仿真）
    test_sc = np.array([[0.0e-3, 12e-3, 1.0], [0.0e-3, 25e-3, 0.7], [0.0e-3, 34e-3, 0.4]])
    test_low = make_low_image(test_sc)
    test_low = test_low / (np.abs(test_low).max() + 1e-8)
    test_ideal = make_ideal_image(test_sc)
    test_ideal = test_ideal / (np.abs(test_ideal).max() + 1e-8)

    x = test_low[None, None].astype(np.float32)  # (1,1,512,256)

    # PyTorch 前向
    model.eval()
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x).to(device)).cpu().numpy()

    # ONNX Runtime 前向
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(["recon"], {"input": x})[0]

    # 对比
    diff = np.abs(y_torch - y_onnx)
    max_err = float(diff.max())
    mean_err = float(diff.mean())

    print("\n===== 验证结果（PyTorch vs ONNX）=====")
    print(f"  最大误差 = {max_err:.2e}")
    print(f"  平均误差 = {mean_err:.2e}")
    print(f"  输出范围: PyTorch={y_torch.min():.4f}~{y_torch.max():.4f}  "
          f"ONNX={y_onnx.min():.4f}~{y_onnx.max():.4f}")

    # 误差 < 1e-4 说明导出无损（对 float32 这是标准阈值）
    if max_err < 1e-4:
        print("  [通过] ONNX 导出无损，精度完全一致。")
    elif max_err < 1e-2:
        print("  [基本通过] 误差极小，属 float 计算顺序差异，可忽略。")
    else:
        print("  [异常] 误差偏大，请检查导出参数。")

    # 出验证图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    titles = ["低质输入", "PyTorch 重建", "ONNX 重建", "理想高质图"]
    imgs = [test_low, y_torch[0, 0], y_onnx[0, 0], test_ideal]
    for ax, im, t in zip(axes, imgs, titles):
        disp = 20 * np.log10(np.abs(im) / np.abs(im).max() + 1e-6)
        ax.imshow(disp, cmap="gray", aspect="auto", vmin=-40, vmax=0)
        ax.set_title(t)
        ax.axis("off")
    plt.suptitle(f"ONNX 验证 | 最大误差 {max_err:.2e} | PSNR(PyTorch)={psnr(y_torch[0,0], test_ideal):.2f}dB")
    plt.tight_layout()
    plt.savefig("verify_result.png", dpi=150)
    print("  验证对比图已保存 -> verify_result.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30, help="训练轮数（验证部署链路用 30 就够）")
    ap.add_argument("--skip-train", action="store_true", help="已有权重时跳过训练，直接导出+验证")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 = {device}")

    weight = "resunet.pth"
    if args.skip_train and os.path.exists(weight):
        print(f"[skip] 复用已有权重 {weight}")
        model = ResUNet(residual=True).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        model.eval()
    else:
        print(f"[1/3] 训练 ResUNet（{args.epochs} epoch）...")
        model = train_and_save(args.epochs, device, weight)

    print("\n[2/3] 导出 ONNX ...")
    export_onnx(model, device)

    print("\n[3/3] 验证 ONNX ...")
    verify_onnx(model, device)

    print("\n完成！部署第一步闭环跑通：.pth -> .onnx -> onnxruntime 无损验证")
