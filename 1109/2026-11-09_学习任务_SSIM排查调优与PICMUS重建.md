# 2026-11-09 学习任务：SSIM 下降排查 + 调优 + 公开数据集（PICMUS）DL 重建

> 状态：待执行（组会 11/11 前最后一个完整工作日；11/10 做 PPT）。按《每日学习定式_v2》交付，助手未执行任何 Python/Notebook/训练/下载。
> 前置：11/6 已完成 DL 重建第一版，结果 **baseline PSNR 24.67 → DL 29.13 dB（+4.46），SSIM 0.8584 → 0.4228（下降）**。Perdios 论文已全部收尾。
> 今日角色：主线「超声重建最小闭环」**收尾冲刺日**——① 搞清楚「为什么 PSNR 升、SSIM 降」（组会必须能解释，否则是硬伤）；② 尝试调优让 SSIM 也回升；③ 把自仿真经验迁移到**公开数据集 PICMUS**，做出组会第二块实证（汇报亮点）。
> 优先级（时间不够就砍后面）：**SSIM 排查 > 调优（至少 1–2 个变式）> PICMUS（能跑最小复现最好，跑不通就如实记卡点）**。

---

## 0. 今日七块（加量版 7–8h，论文已收尾故压缩）

| 顺序 | 块 | 时间 | 必须产出 |
|---|---|---|---|
| 1 | 开场串联＋11/6 复盘 | 10min | 三问答案＋3 抽问＋1 重述 |
| 2 | 主任务 A：SSIM 下降排查 | 2.5h | 口径自检＋三分量分解＋「哪个成分降了」的结论 |
| 3 | 主任务 B：调优 | 1.5–2h | 至少 2 个变式，PSNR/SSIM 对比表 |
| 4 | 主任务 C：PICMUS DL 重建 | 2h（可降级） | RF→DAS→配对→重建（或如实记录卡点） |
| 5 | 前沿摘要泛读 | 0（压缩，规则 23 允许） | 跳过 |
| 6 | 今日复盘 | 10–15min | 3 抽问＋1 重述＋标记 |
| 7 | 日终记录 | 10min | 三行＋填报告骨架 |

> **今天为项目贡献什么**：① 把「SSIM 下降」从「异常待查」变成「可解释的结论」——这是组会上导师一定会问的，答不上来比没做更糟；② 若调优成功，得到「PSNR + SSIM 双升」的干净数字；③ PICMUS 真实 RF 数据重建成功，证明方法不是只在自仿真数据上成立。

---

## 1. 开场串联＋11/6 复盘（10 分钟）

### 1.1 开场三问

1. **昨天（11/6）学了什么？今天和它什么关系？**
   昨天：把「单 PW 低质 / 5 PW 高质」一对图放大成 200 对，训了个逐像素回归网络，PSNR 升了但 SSIM 降了。今天：先搞清 SSIM 为什么降，再想办法让它也升，最后把同一套方法搬到真实公开数据 PICMUS 上。
2. **今天这块在整条项目链的哪个位置？**
   `数据 → 波束形成 → 低质/高质配对 → DL 重建 → [今天：指标可信性 + 调优] → baseline 对比`。今天是把「第一版能跑」升级成「能讲清楚 + 能站得住脚」的收尾一步，同时把「仿真数据」这条腿换成「公开真实数据」。
3. **今天学完，我能多做哪一件具体的事？**
   能：① 说出「PSNR 升 + SSIM 降」背后的机理（不是背结论）；② 用 SSIM 三分量（亮度/对比度/结构）定位是「过平滑」还是「口径 bug」；③ 把一个自仿真 DAS 重建脚本改到真实 RF 数据（PICMUS）上跑通。

### 1.2 11/6 复盘（闭卷先答）

| 抽问 | 参考答案 |
|---|---|
| ① 逐像素回归的输出层/loss/评价各是什么？ | Conv2d(→1) 无激活 / MSE / PSNR+SSIM |
| ② baseline 为什么取「输入低质图」本身？ | 不重建的最低标准，DL 必须超过它才证明重建有用 |
| ③ PSNR 和 MSE 什么关系？ | PSNR=10·log₁₀(MAX²/MSE)，MSE 的单调函数，MSE 越小 PSNR 越大 |

**重述（一句话）**：昨天证明了「DL 重建在 PSNR 上超过 baseline」；今天要补齐「SSIM 为什么反而降」，并把方法迁移到真实数据。

---

## 2. 主任务 A：SSIM 下降排查（2.5 小时，第一优先级）

### 2.0 先定义：为什么「PSNR 升 + SSIM 降」是典型现象，不是必然 bug（规则 2：先定义后上手）

**MSE 的最优解是条件均值。** 用 MSE 训练「低质→高质」回归，网络学到的输出会趋近 **E[高质 | 低质]**——也就是「同一批低质图对应的所有高质图的平均」。而超声里的 **speckle（斑点纹理）是随机乘性噪声**，同一低质图可以对应多种 speckle 图案的高质图，取平均就会把 speckle「抹平」，输出变平滑。

| 指标 | 看什么 | 平滑的影响 |
|---|---|---|
| PSNR | 逐像素误差（MSE 的单调函数） | 平滑让像素更接近平均目标 → **升** |
| SSIM | 亮度/对比度/结构 | 平滑降低对比度（方差↓）、改变结构（纹理丢）→ **降** |

**一句话**：「用 MSE 训练、用 SSIM 评价」是**指标错配**；「PSNR 升 + SSIM 降」正是这个错配的典型信号。但**下结论前必须先排除两件事**：① SSIM 实现/口径有 bug（算错了）；② 归一化 clip 把信息压没了。所以今天排查分两条线。

> ⚠️ 诚实边界：0.8584 → 0.4228 跌幅近半，比「轻微过平滑」更大，所以**不能只靠机理解释糊弄过去**，必须用下面的诊断代码定位到底发生了什么。

### 2.1 第 1 步：SSIM 自检（自己写，先排除口径 bug）

SSIM 函数正确性的硬标准：**一张图和它自己比，SSIM 必须 = 1.0**（亮度/对比度/结构三者完全一致）。先跑这个自检：

```python
# 复用 11/6 的 ssim（高斯窗口 11x11, sigma=1.5, C1=1e-4, C2=9e-4）
print("自检 ssim(Xte, Xte) =", ssim(Xte, Xte))   # 应 ≈ 1.0（允许 0.99x 的浮点误差）
print("自检 ssim(Yte, Yte) =", ssim(Yte, Yte))   # 应 ≈ 1.0
print("自检 ssim(全0, 全0) =", ssim(torch.zeros_like(Yte), torch.zeros_like(Yte)))
# 注意：两张全零图 SSIM 可能报 NaN 或异常（C1/C2 兜底），若如此说明实现要加 1e-12 保护
```

**跑完要能回答**：① 自检是否 = 1.0？如果不是，SSIM 函数本身有 bug，先修函数再往下。② 两张全零图会怎样？为什么？（分母只有 C1/C2 兜底，可能 NaN）

### 2.2 第 2 步：把 SSIM 拆成三分量，定位「哪个成分降了」（自己写，今日诊断核心）

SSIM = **亮度 l × 对比度 c × 结构 s** 三个分量相乘。拆开看，就知道是「过平滑（c 降）」还是「结构变了（s 降）」还是「亮度漂移（l 降）」：

```python
import torch
import torch.nn.functional as F

def gaussian_kernel(size=11, sigma=1.5):
    c = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(c**2) / (2*sigma**2)); g = g / g.sum()
    k = g[:, None] * g[None, :]
    return k.view(1, 1, size, size)
_W = gaussian_kernel()

def ssim_3comp(a, b, C1=1e-4, C2=9e-4):
    """返回亮度 l、对比度 c、结构 s 三分量的均值（乘积 ≈ 整图 SSIM）。"""
    w = _W.to(a.device); pad = 5
    mu1 = F.conv2d(a, w, padding=pad)
    mu2 = F.conv2d(b, w, padding=pad)
    s1 = F.conv2d(a*a, w, padding=pad) - mu1**2
    s2 = F.conv2d(b*b, w, padding=pad) - mu2**2
    s12 = F.conv2d(a*b, w, padding=pad) - mu1*mu2
    sig1 = torch.sqrt(s1.clamp(min=0) + 1e-12)
    sig2 = torch.sqrt(s2.clamp(min=0) + 1e-12)
    l = (2*mu1*mu2 + C1) / (mu1**2 + mu2**2 + C1)          # 亮度
    c = (2*sig1*sig2 + C2) / (s1 + s2 + C2)               # 对比度
    s = (s12 + C2/2) / (sig1*sig2 + C2/2)                 # 结构
    return l.mean().item(), c.mean().item(), s.mean().item()

# baseline（低质 vs 高质）和 DL（重建 vs 高质）各拆一次
for name, x in [("baseline", Xte), ("DL重建", pred)]:
    l, c, s = ssim_3comp(x, Yte)
    print(f"{name}: 亮度l={l:.4f} 对比度c={c:.4f} 结构s={s:.4f} 乘积={l*c*s:.4f}")
```

**跑完要能回答**：① DL 重建相对 baseline，是**哪个分量**掉得最狠？（预期：对比度 c，说明过平滑）② 三个分量的乘积 ≈ 之前的整图 SSIM 吗？（应大致一致，不一致说明口径有出入，回头对齐窗口/常数）

### 2.3 第 3 步：用「标准差」和「直方图」直接看过平滑（变式，先预测再运行）

三分量里对比度 c 掉 = DL 输出的方差比目标小。用最直观的方式验证：

```python
print("低质输入 std:", Xte.std().item())
print("高质目标 std:", Yte.std().item())
print("DL 输出  std:", pred.std().item())   # 预测：明显 < 高质目标 → 过平滑
```

再画直方图对比灰度分布（过平滑 = 直方图变窄、往中间挤）：

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
i = 0
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow((Yte[i,0] - pred[i,0]).cpu(), cmap="bwr", vmin=-0.3, vmax=0.3, aspect="auto")
ax[0].set_title("残差 Yte - pred（红=DL偏暗，蓝=DL偏亮）")
ax[1].hist(Yte[i,0].cpu().flatten(), bins=60, alpha=0.5, label="高质目标")
ax[1].hist(pred[i,0].cpu().flatten(), bins=60, alpha=0.5, label="DL 输出")
ax[1].legend(); ax[1].set_title("灰度直方图（过平滑→DL 更窄更集中）")
plt.show()
```

**验收（A 部分的结论，写进日终）**：三条证据是否互相印证——① 自检通过（函数没 bug）；② 对比度 c 掉（或结构 s 掉）；③ std 变小 + 直方图变窄。若三条都指向「过平滑」，机理结论成立；若对不上（如自检都不过、或 l 掉很多），说明是口径/归一化问题，转入 2.4 排查。

### 2.4 备用排查：归一化口径（如果自检不过、或 l 分量异常才做）

11/6 的 `norm(x) = clip((x+60)/60, 0, 1)` 把 dB 图 [−60,0] 压到 [0,1]。注意两点可能引入问题：
- **clip 截断**：低于 −60 dB 的像素被一刀切到 0，把大量背景细节抹平；
- **逐图各自 `d.max()` 归一**：每张图按自己的最大值缩放，动态范围尺度不一致。

排查：打印归一化前 `disp_db` 输出中「被 clip 掉的比例」：
```python
lo, hi = make_pair()
db = lo  # 或 hi
print("clip 到 0 的像素比例:", (db < -60).mean())   # 若 > 0.2，动态范围定窄了
print("db 实际范围:", db.min(), db.max())
```

---

## 3. 主任务 B：调优（1.5–2 小时，第二优先级）

目标：在**不破坏 PSNR 提升**的前提下，把 SSIM 拉回来。三个变式，**先预测再运行**，最后填一张对比表。

> 保持规则 19（基线优先 + 同口径）：每个变式都重训、都用**同一批 Xte/Yte、同一个 ssim 函数**对比，否则数字不可比。

### 3.1 变式 A：换 L1 loss（自己写）

MSE 平方会惩罚大误差，更倾向「抹平均」；**L1 更保边缘、不过度平滑**。只改一行：

```python
loss_fn = nn.L1Loss()   # 原来 nn.MSELoss()，其余训练代码不动
```

**预测**：L1 相比 MSE，PSNR 可能略降（L1 不是最优 MSE 解），但 SSIM 应回升（保纹理）。

### 3.2 变式 B：组合损失 MSE + λ·(1−SSIM)（自己写，今日调优重点）

让网络在训练时**同时**优化像素误差和结构相似度。给一个可导的简化 SSIM（用平均池化代替高斯卷积，保证能反向传播）：

```python
def ssim_map(a, b, C1=1e-4, C2=9e-4, win=11):
    mu1 = F.avg_pool2d(a, win, 1, win//2)
    mu2 = F.avg_pool2d(b, win, 1, win//2)
    s1  = F.avg_pool2d(a*a, win, 1, win//2) - mu1**2
    s2  = F.avg_pool2d(b*b, win, 1, win//2) - mu2**2
    s12 = F.avg_pool2d(a*b, win, 1, win//2) - mu1*mu2
    num = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return num / (den + 1e-8)

def combined_loss(pred, target, lam=0.1):
    return F.mse_loss(pred, target) + lam * (1 - ssim_map(pred, target).mean())
```

训练时用 `loss = combined_loss(pred, Ytr)` 代替 `loss_fn(pred, Ytr)`。可试 lam = 0.05 / 0.1 / 0.2 三档。

**预测**：lam 越大越保结构，SSIM 升，但 PSNR 可能从 29 dB 略降；找一个「两者都好」的平衡点。

### 3.3 变式 C：归一化口径（只做如果 2.4 发现 clip 比例高）

若 clip 比例高，把动态范围放宽（如 [−80, 0]）或对整批用统一 min/max 归一，重训看 SSIM。

### 3.4 调优对比表（日终必填）

| 方案 | PSNR (dB) | SSIM | 相对 baseline 的 PSNR/SSIM 变化 |
|---|---|---|---|
| baseline（输入低质图） | 24.67 | 0.8584 | 0 / 0 |
| 11/6 MSE（第一版） | 29.13 | 0.4228 | +4.46 / −0.44 |
| 变式 A：L1 | ? | ? | |
| 变式 B：MSE+λSSIM（记 λ） | ? | ? | |

> 目标：找到一行「PSNR 保持 +4 dB 以上，且 SSIM 不低于 baseline（或至少大幅回升）」的方案，作为组会上的干净结果；如果实在调不出，就如实讲「MSE 过平滑 + SSIM 指标错配」这个现象本身——这同样是一个能讲、且显得你懂指标的结论。

---

## 4. 主任务 C：公开数据集（PICMUS）DL 重建（2 小时，第三优先级，可降级）

### 4.0 PICMUS 是什么（先定义）

**PICMUS（Plane-wave Imaging Challenge in Medical UltraSound，IEEE IUS 2016）**：一个公开的平面波超声成像挑战赛数据集，提供**真实 RF 通道数据**（in-vitro 实验 phantom + in-vivo 颈动脉），是超声重建领域公认的公开 benchmark。

> **为什么不自己造了？** 自仿真点散射体是「干净但理想」的数据；PICMUS 是「真实探头采集」的数据，用它跑通 DL 重建，能证明你的方法不是只在玩具数据上成立——这是组会汇报的第二块实证（亮点）。

**数据获取（你自己下载，助手不下载）**：
- HuggingFace 镜像：`https://huggingface.co/datasets/zeahub/picmus`
- 官方挑战页：IEEE IUS 2016 平面波成像挑战（CREATIS INSA Lyon）——以官方 README 为准。

> ⚠️ **诚实边界 + 关键纪律**：PICMUS 数据格式、阵元数 N、采样率 fs、平面波角度数组 angles、通道/帧维度**都以你下载到的 README 为准，不要照抄下面代码里的默认值**。下载后第一步是**打印 `rf.shape` 和读 README 的参数表**，把真实参数填进变量，再跑 DAS。

### 4.1 第 1 步：读数据 + 打印形状（自己写）

```python
import numpy as np
# 按 README 的加载方式读入一帧 RF（具体文件名/函数以数据为准）
# rf = ...   # 预期形状类似 (N_angles, N_elements, T) 或 (N_elements, T, N_angles)
print("rf.shape =", rf.shape)
# 打印参数
N_elements = ...   # 阵元数（README）
fs         = ...   # 采样率（README）
pitch      = ...   # 阵元间距（README）
angles     = ...   # 平面波角度数组（弧度，README）
print(N_elements, fs, pitch, angles.shape, angles[[0, -1]])
```

### 4.2 第 2 步：RF → DAS（复用 11/5 的手写 DAS，只改参数，自己写）

发射是平面波（发射延迟 = `x·sinθ`），接收是逐阵元时延求和。**和 11/5 自仿真几乎一样**，只是参数换成 PICMUS 真实值：

```python
c = 1540.0
x_elem = (np.arange(N_elements) - (N_elements - 1) / 2) * pitch
T = rf.shape[-1]
t_axis = np.arange(T) / fs
z_max = t_axis[-1] * c / 2     # 往返 → 深度折半

def das_picmus(rf_angle, theta, nx=256, nz=512, z0=0.0, z1=None):
    """rf_angle: 单个平面波角度的 RF (N_elements, T)。"""
    z1 = z_max if z1 is None else z1
    x_grid = np.linspace(x_elem.min(), x_elem.max(), nx)
    z_grid = np.linspace(z0, z1, nz)
    X, Z = np.meshgrid(x_grid, z_grid)
    img = np.zeros((nz, nx))
    for i in range(N_elements):
        tau = (X*np.sin(theta) + Z*np.cos(theta))/c \
            + np.sqrt((X - x_elem[i])**2 + Z**2)/c      # 发射 + 接收时延
        idx = np.clip(np.round(tau*fs).astype(int), 0, T-1)
        img += rf_angle[i, idx]
    return img, x_grid, z_grid

def disp_db(img):
    d = np.abs(img)
    return 20*np.log10(d/(d.max() + 1e-12) + 1e-6)
```

**低质 = 单平面波（如 0°）**，**高质 = 所有角度相干复合**：

```python
th0 = 0.0
low, xg, zg = das_picmus(rf[angle_idx_of_0deg], th0)      # 低质：1 个角度
high = np.zeros_like(low)
for k, th in enumerate(angles):
    high += das_picmus(rf[k], th)[0]                       # 高质：所有角度复合
high /= len(angles)
low_db, high_db = disp_db(low), disp_db(high)
```

**跑完要能回答**：① 这里 DAS 和 11/5 自仿真的区别是什么？（发射侧：自仿真用点散射体解析式造 RF，这里用真实采集 RF；延迟公式本质相同）② 为什么「所有角度复合」是高质量？（合成大孔径、抑制旁瓣，等价于 11/6 的相干复合）

### 4.3 第 3 步：配对 patch + DL 重建（复用 11/6 训练代码，自己写）

PICMUS 是**静态 phantom，不能像自仿真那样随机造几百对**。配对方式：从同一帧的 low_db/high_db 上**切出很多小块（patch）**当训练样本：

```python
def extract_patches(img, size=64, stride=32):
    patches = []
    h, w = img.shape
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            patches.append(img[r:r+size, c:c+size])
    return np.stack(patches)

P_low  = extract_patches(low_db)    # (K, size, size)
P_high = extract_patches(high_db)
# 归一化（同一 norm 函数，口径一致）→ 转 tensor (K,1,size,size) → 复用 11/6 的 ReconNet + 训练 + psnr/ssim
```

**跑完要能回答**：① 为什么 PICMUS 用「切 patch」造配对，而自仿真用「随机散射体」造配对？（PICMUS 场景固定，只能靠空间切块扩充样本）② patch 之间有重叠（stride<size）会怎样？（样本不独立、可能过拟合到个别块，可接受但要知道）

### 4.4 验收 + 降级路径

- [ ] 打印出 PICMUS 真实 rf.shape 和参数；
- [ ] 低质/高质 B 图各出一张（肉眼：单角度糊、多角度清晰）；
- [ ] DL 重建 + PSNR/SSIM 数字（同口径对比 baseline）；
- [ ] 若数据格式/下载卡住：**如实记录卡点**（缺什么、卡在哪），不要硬凑数字，组会讲「自仿真已闭环 + PICMUS 已到 X 步」。

---

## 5. 前沿摘要泛读（压缩，规则 23 允许）

今天时间全砸主任务，本块跳过。

---

## 6. 今日复盘（10–15 分钟）

| 抽问 | 参考答案 |
|---|---|
| ① MSE 训练为什么倾向「过平滑」？ | MSE 最优解是条件均值 E[高质\|低质]，把随机 speckle 平均掉 |
| ② SSIM 三分量分别是什么？过平滑主要掉哪个？ | 亮度 l / 对比度 c / 结构 s；过平滑主要掉对比度 c |
| ③ 为什么「PSNR 升 + SSIM 降」要警惕但不能立刻当 bug？ | 是「MSE 训练 + SSIM 评价」的指标错配典型信号，但须先排除口径/归一化 bug |

**重述（今日最关键概念，用自己的话）**：PSNR 和 SSIM 看的东西不同——PSNR 只看逐像素差多少，SSIM 看结构和纹理像不像；用 MSE 训练会得到「最平均、最平滑」的解，所以 PSNR 涨、SSIM 掉。要么换/组合 loss 把结构找回来，要么把它讲成一个「你懂指标错配」的结论。

标记：已掌握 / 仍模糊分别列出。

---

## 7. 日终记录（10 分钟）

```text
实际有效用时：            开始/结束时间：
完成到第几块：
开场三问答案（各一句）：
11/6 复盘：抽问对几题、仍模糊：

主任务 A（SSIM 排查）：
  - SSIM 自检是否 = 1.0？
  - 三分量：baseline 和 DL 各 l/c/s 是多少？哪个分量掉？
  - std/直方图是否印证过平滑？clip 像素比例？
  - 结论：过平滑 / 口径bug / 归一化问题（三选一，附证据）

主任务 B（调优）：变式 A/B/C 各 PSNR/SSIM 填入 3.4 表，预测 vs 实际，最佳方案

主任务 C（PICMUS）：数据是否下载？rf.shape 和参数？低/高质 B 图是否出？
  DL 重建 PSNR/SSIM 数字？还是卡点（如实写缺什么、卡在哪）

小测 3 题对几题：
文件实际路径（数据 .npy / 权重 / 图 / 笔记本）：
未完成项与对 11/10（PPT）排期的影响：
```

---

## 8. 下一步（11/10 周二）

- 11/10：**用户自己做 PPT**（组会材料 + 讲稿演练，助手不代做内容；汇报时长/组织由用户定）。
- 11/11：组会讲项目 + Perdios 收尾汇报。
