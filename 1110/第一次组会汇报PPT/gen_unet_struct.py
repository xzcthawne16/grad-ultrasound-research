# -*- coding: utf-8 -*-
"""
重新生成 UNet 结构图 —— 字体放大版（字号整体调大 20 号）
输出：assets/unet_struct.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 字号基准（原图约 12~14 号，这里整体调大到 32~34 号，即 +20 号）
FS_TITLE = 34      # 模块标题（enc1 / bottleneck 等）
FS_SUB   = 26      # 通道·尺寸标注
FS_INOUT = 24      # 输入/输出标签
FS_SKIP  = 24      # 跳跃连接标注
FS_LEG   = 22      # 图例文字

# 颜色
C_ENC  = '#4A90D9'   # 编码器蓝
C_DEC  = '#4CAF50'   # 解码器绿
C_BOT  = '#9B59B6'   # 瓶颈紫
C_SKIP = '#9E9E9E'   # 跳跃连接灰

fig, ax = plt.subplots(figsize=(14, 9), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

def box(x, y, w, h, fc, ec, lw=2):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    return b

def arrow(x1, y1, x2, y2, color, lw=2.2, style='-|>', ls='-'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=22, linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)

# ---- 坐标规划（横向两列，纵向 5 层 + 底部瓶颈）----
# 编码器列 x=3.0，解码器列 x=11.0，瓶颈 x=7.0, y=1.0
# 层 y 从上到下：输入 8.0, enc1 6.6, enc2 5.2, enc3 3.8, enc4 2.4, 瓶颈 1.0
# 解码器（从下到上）：dec4 2.4, dec3 3.8, dec2 5.2, dec1 6.6, 输出 8.0

BW, BH = 2.0, 0.9   # 模块框宽高

enc_layers = [
    ('enc1', '32 通道 · 64×64', 6.6),
    ('enc2', '64 通道 · 32×32', 5.2),
    ('enc3', '128 通道 · 16×16', 3.8),
    ('enc4', '256 通道 · 8×8', 2.4),
]
dec_layers = [
    ('dec4', '256 通道 · 8×8', 2.4),
    ('dec3', '128 通道 · 16×16', 3.8),
    ('dec2', '64 通道 · 32×32', 5.2),
    ('dec1', '32 通道 · 64×64', 6.6),
]

# 输入/输出（框加高，避免文字重叠）
BIO_H = 1.25
box(3.0, 8.0, BW, BIO_H, '#E3F2FD', C_ENC)
ax.text(3.0, 8.25, '输入 低质图', ha='center', va='center', fontsize=FS_INOUT, color='#1A237E', fontweight='bold')
ax.text(3.0, 7.72, '1×64×64', ha='center', va='center', fontsize=FS_SUB, color='#37474F')

box(11.0, 8.0, BW, BIO_H, '#E8F5E9', C_DEC)
ax.text(11.0, 8.25, '输出 重建图', ha='center', va='center', fontsize=FS_INOUT, color='#1B5E20', fontweight='bold')
ax.text(11.0, 7.72, '1×64×64', ha='center', va='center', fontsize=FS_SUB, color='#37474F')

# 编码器模块
for name, sub, y in enc_layers:
    box(3.0, y, BW, BH, '#E3F2FD', C_ENC)
    ax.text(3.0, y + 0.14, name, ha='center', va='center', fontsize=FS_TITLE, color='#1565C0', fontweight='bold')
    ax.text(3.0, y - 0.22, sub, ha='center', va='center', fontsize=FS_SUB, color='#37474F')

# 解码器模块
for name, sub, y in dec_layers:
    box(11.0, y, BW, BH, '#E8F5E9', C_DEC)
    ax.text(11.0, y + 0.14, name, ha='center', va='center', fontsize=FS_TITLE, color='#2E7D32', fontweight='bold')
    ax.text(11.0, y - 0.22, sub, ha='center', va='center', fontsize=FS_SUB, color='#37474F')

# 瓶颈
box(7.0, 1.0, BW + 0.6, BH, '#F3E5F5', C_BOT)
ax.text(7.0, 1.14, 'bottleneck', ha='center', va='center', fontsize=FS_TITLE, color='#6A1B9A', fontweight='bold')
ax.text(7.0, 0.78, '512 通道 · 4×4', ha='center', va='center', fontsize=FS_SUB, color='#37474F')

# ---- 箭头 ----
# 输入 -> enc1（输入框下缘在 8.0 - BIO_H/2 = 7.375）
arrow(3.0, 8.0 - BIO_H/2, 3.0, 6.6 + BH/2, C_ENC)
# 编码器纵向
for i in range(len(enc_layers) - 1):
    arrow(3.0, enc_layers[i][2] - BH/2, 3.0, enc_layers[i+1][2] + BH/2, C_ENC)
# enc4 -> 瓶颈
arrow(3.0, 2.4 - BH/2, 7.0 - BW/2 - 0.3, 1.0, C_ENC)
# 瓶颈 -> dec4
arrow(7.0 + BW/2 + 0.3, 1.0, 11.0, 2.4 - BH/2, C_DEC)
# 解码器纵向（从下到上）
for i in range(len(dec_layers) - 1):
    arrow(11.0, dec_layers[i][2] + BH/2, 11.0, dec_layers[i+1][2] - BH/2, C_DEC)
# dec1 -> 输出（输出框下缘在 7.375）
arrow(11.0, 6.6 + BH/2, 11.0, 8.0 - BIO_H/2, C_DEC)

# ---- 跳跃连接（虚线）----
skip_pairs = [(6.6, 6.6), (5.2, 5.2), (3.8, 3.8)]
for y_enc, y_dec in skip_pairs:
    arrow(3.0 + BW/2, y_enc, 11.0 - BW/2, y_dec, C_SKIP, ls='--', lw=1.8)

# 跳跃连接标注（放在中间，dec3 层附近）
ax.text(7.0, 3.75, '跳跃连接', ha='center', va='center', fontsize=FS_SKIP,
        color='#616161', style='italic')

# ---- 图例 ----
ax.text(2.5, 0.28, '■ 蓝 = 编码器（下采样）', ha='left', va='center',
        fontsize=FS_LEG, color=C_ENC, fontweight='bold')
ax.text(7.0, 0.28, '■ 紫 = 瓶颈', ha='center', va='center',
        fontsize=FS_LEG, color=C_BOT, fontweight='bold')
ax.text(11.5, 0.28, '■ 绿 = 解码器（上采样）', ha='right', va='center',
        fontsize=FS_LEG, color=C_DEC, fontweight='bold')

plt.tight_layout(pad=0.5)
out = r'E:\develop\Graduate_AI\2026-11\1110\组会汇报PPT\assets\unet_struct.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print('saved:', out)

from PIL import Image
im = Image.open(out)
print('size:', im.size)
