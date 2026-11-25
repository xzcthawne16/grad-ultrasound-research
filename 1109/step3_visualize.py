# 主任务A第3步：可视化验证（低质输入 vs DL重建 vs 高质目标）
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 配置中文字体，避免方块乱码
for f in ['SimHei', 'Microsoft YaHei', 'Noto Sans SC']:
    try:
        font_manager.fontManager.addfont(font_manager.findfont(f, fallback_to_default=False))
        plt.rcParams['font.sans-serif'] = [f, 'SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue
import time

c, fs, N, pitch, sigma = 1540.0, 40e6, 64, 0.3e-3, 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch
t_axis = np.arange(0, 2 * 40e-3 / c, 1 / fs)
angles = np.deg2rad(np.array([-5, -2.5, 0, 2.5, 5]))

nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx).astype(np.float32)
z_grid = np.linspace(2e-3, 40e-3, nz).astype(np.float32)
X, Z = np.meshgrid(x_grid, z_grid)

cache = {}
for th in angles:
    sin = np.sin(th); cos = np.cos(th)
    d = (X[:, :, None] * sin + Z[:, :, None] * cos) / c \
        + np.sqrt((X[:, :, None] - elem_x[None, None, :].astype(np.float32)) ** 2 + Z[:, :, None] ** 2) / c
    cache[th] = d.astype(np.float32)

def rand_sc(n):
    return np.array([(np.random.uniform(-8e-3, 8e-3),
                      np.random.uniform(5e-3, 35e-3),
                      np.random.uniform(0.3, 1.0)) for _ in range(n)])

def make_rf_vec(theta, sc):
    xk = sc[:, 0]; zk = sc[:, 1]; amp = sc[:, 2]
    tau = (xk[:, None] * np.sin(theta) + zk[:, None] * np.cos(theta)) / c \
        + np.sqrt((xk[:, None] - elem_x[None, :]) ** 2 + zk[:, None] ** 2) / c
    env = amp[:, None, None] * np.exp(-((t_axis[None, None, :] - tau[:, :, None]) ** 2) / (2 * sigma ** 2))
    return env.sum(axis=0)

def das_fast(rf, th):
    idx = np.clip(np.round(cache[th] * fs).astype(np.int64), 0, len(t_axis) - 1)
    return rf[np.arange(N), idx].sum(axis=2)

def disp_db(img):
    d = np.abs(img)
    return 20 * np.log10(d / d.max() + 1e-6)

def make_pair():
    sc = rand_sc(np.random.randint(3, 8))
    lo = das_fast(make_rf_vec(0.0, sc), 0.0)
    hi = np.zeros_like(lo)
    for th in angles:
        hi += das_fast(make_rf_vec(th, sc), th)
    hi /= len(angles)
    return disp_db(lo), disp_db(hi)

torch.manual_seed(0); np.random.seed(0)
Ntrain, Ntest = 160, 40
t0 = time.time()
Xtr, Ytr = [], []
for _ in range(Ntrain):
    lo, hi = make_pair(); Xtr.append(lo); Ytr.append(hi)
Xte, Yte = [], []
for _ in range(Ntest):
    lo, hi = make_pair(); Xte.append(lo); Yte.append(hi)
print(f"[数据] {time.time()-t0:.0f}s", flush=True)

def norm(x):
    return np.clip((x + 60) / 60.0, 0, 1)
Xtr = torch.tensor(norm(np.stack(Xtr))[:, None], dtype=torch.float32)
Ytr = torch.tensor(norm(np.stack(Ytr))[:, None], dtype=torch.float32)
Xte = torch.tensor(norm(np.stack(Xte))[:, None], dtype=torch.float32)
Yte = torch.tensor(norm(np.stack(Yte))[:, None], dtype=torch.float32)

class ReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
    def forward(self, x): return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ReconNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=16, shuffle=True)

for ep in range(60):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    pred = model(Xte.to(device)).cpu()

# ===== 可视化：画 3 张测试样本的对比 =====
fig, axes = plt.subplots(3, 3, figsize=(14, 14))
titles = ['低质输入 (1PW)', 'DL 重建', '高质目标 (5PW)']

for i in range(3):
    x_in = Xte[i, 0].numpy()
    y_out = pred[i, 0].numpy()
    y_gt = Yte[i, 0].numpy()
    imgs = [x_in, y_out, y_gt]
    for j in range(3):
        ax = axes[i, j]
        im = ax.imshow(imgs[j], cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'样本{i+1} {titles[j]}', fontsize=11)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('DL 重建效果对比：低质 → 重建 → 高质', fontsize=15, y=0.995)
plt.tight_layout()
out = 'E:/develop/Graduate_AI/2026-11/1109/step3_visualize.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'已保存对比图: {out}', flush=True)

# 同时打印这三张的逐样本 SSIM 参考
import torch.nn.functional as F
def gaussian_kernel(size=11, sigma=1.5):
    cc = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(cc**2) / (2*sigma**2)); g = g / g.sum()
    k = g[:, None] * g[None, :]
    return k.view(1, 1, size, size)
_W = gaussian_kernel()
def ssim(a, b, C1=1e-4, C2=9e-4):
    w = _W.to(a.device); pad = 5
    mu1 = F.conv2d(a, w, padding=pad); mu2 = F.conv2d(b, w, padding=pad)
    s1 = F.conv2d(a*a, w, padding=pad) - mu1**2
    s2 = F.conv2d(b*b, w, padding=pad) - mu2**2
    s12 = F.conv2d(a*b, w, padding=pad) - mu1*mu2
    m = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return m.mean().item()

print("\n逐样本 SSIM（前3个测试样本）:")
for i in range(3):
    s_base = ssim(Xte[i:i+1], Yte[i:i+1])
    s_dl = ssim(pred[i:i+1], Yte[i:i+1])
    print(f"  样本{i+1}: baseline={s_base:.4f}  DL重建={s_dl:.4f}")
