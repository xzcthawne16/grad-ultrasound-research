# -*- coding: utf-8 -*-
"""生成「近期工作汇报」PPT（10 页，简洁版，面向导师单独汇报）。"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

FONT = "Microsoft YaHei"
NAVY = RGBColor(0x0F, 0x2B, 0x46)
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
GREY = RGBColor(0x5A, 0x6B, 0x7B)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
LIGHT = RGBColor(0xF3, 0xF6, 0xFB)
LINE = RGBColor(0xD8, 0xE1, 0xEC)
GREEN = RGBColor(0x1E, 0x8E, 0x3E)
AMBER = RGBColor(0xB2, 0x6A, 0x00)

W, H = 13.333, 7.5
BASE = Path(r"E:\develop\Graduate_AI")


def set_run(run, size=16, bold=False, color=DARK, name=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = name
    rPr = run._r.get_or_add_rPr()
    latin = rPr.find(qn("a:latin"))
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        if latin is not None:
            latin.addnext(ea)
        else:
            rPr.append(ea)
    ea.set("typeface", name)


def textbox(slide, left, top, width, height, text="", size=16, bold=False,
            color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    r = p.add_run()
    r.text = text
    set_run(r, size=size, bold=bold, color=color)
    return box


def bullets(slide, left, top, width, height, items, size=16, color=DARK,
            bullet_color=ACCENT, gap=6, line_spacing=1.15):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            head, detail = item
        else:
            head, detail = item, None
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(gap)
        r0 = p.add_run()
        r0.text = "▪  "
        set_run(r0, size=size, bold=True, color=bullet_color)
        r1 = p.add_run()
        r1.text = head
        set_run(r1, size=size, bold=(detail is not None), color=color)
        if detail:
            r2 = p.add_run()
            r2.text = "  " + detail
            set_run(r2, size=size - 1, bold=False, color=GREY)
    return box


def accent_bar(slide):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(0.09))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False


def frame(slide):
    accent_bar(slide)


def title(slide, text, kicker=None):
    if kicker:
        textbox(slide, 0.7, 0.28, 12, 0.35, kicker, size=12, bold=True, color=ACCENT)
        textbox(slide, 0.7, 0.58, 12, 0.62, text, size=27, bold=True, color=NAVY)
        y = 1.28
    else:
        textbox(slide, 0.7, 0.42, 12, 0.66, text, size=27, bold=True, color=NAVY)
        y = 1.14
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(y), Inches(1.1), Inches(0.05))
    ln.fill.solid()
    ln.fill.fore_color.rgb = ACCENT
    ln.line.fill.background()
    ln.shadow.inherit = False


def footer(slide, idx):
    textbox(slide, 0.7, 7.05, 8, 0.3, "近期工作汇报 · 2026-11-03", size=10, color=GREY)
    textbox(slide, 11.6, 7.05, 1.0, 0.3, f"{idx:02d}", size=10, color=GREY, align=PP_ALIGN.RIGHT)


def card(slide, left, top, w, h, head, body, accent=ACCENT, head_size=15, body_size=12.5):
    r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(w), Inches(h))
    r.adjustments[0] = 0.06
    r.fill.solid()
    r.fill.fore_color.rgb = LIGHT
    r.line.color.rgb = LINE
    r.line.width = Pt(1)
    r.shadow.inherit = False
    tag = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.07), Inches(h))
    tag.fill.solid()
    tag.fill.fore_color.rgb = accent
    tag.line.fill.background()
    tag.shadow.inherit = False
    textbox(slide, left + 0.18, top + 0.1, w - 0.3, 0.34, head, size=head_size, bold=True, color=NAVY)
    box = slide.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.46),
                                   Inches(w - 0.32), Inches(h - 0.54))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (h_, d_) in enumerate(body):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        p.space_before = Pt(3)
        r0 = p.add_run()
        r0.text = "▪ "
        set_run(r0, size=11, bold=True, color=accent)
        r1 = p.add_run()
        r1.text = h_
        set_run(r1, size=body_size, bold=False, color=DARK)
        if d_:
            p2 = tf.add_paragraph()
            p2.line_spacing = 1.05
            r2 = p2.add_run()
            r2.text = "   " + d_
            set_run(r2, size=body_size - 1.2, bold=False, color=GREY)


def table(slide, header, rows, left, top, width, col_widths, row_h=0.42,
          head_size=13, body_size=13, highlight_rows=None):
    n_rows = len(rows) + 1
    n_cols = len(header)
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                   Inches(width), Inches(row_h * n_rows))
    tbl = shape.table
    for j, cw in enumerate(col_widths):
        tbl.columns[j].width = Inches(cw)
    for i in range(n_rows):
        tbl.rows[i].height = Inches(row_h)
    for j, h in enumerate(header):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.margin_left = Inches(0.08)
        cell.margin_right = Inches(0.05)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        set_run(r, size=head_size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i % 2 else LIGHT
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j != 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            bold = highlight_rows is not None and i in highlight_rows
            set_run(r, size=body_size, bold=bold, color=DARK)
    return tbl


def add_image_fit(slide, path, left, top, max_w, max_h):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    l = left + (max_w - w) / 2
    t = top + (max_h - h) / 2
    slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))
    return w, h


# ============================ 构建 ============================
prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]

# ---------- 1 封面 ----------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
bg.shadow.inherit = False
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(4.62), Inches(W), Inches(0.06))
band.fill.solid()
band.fill.fore_color.rgb = ACCENT
band.line.fill.background()
band.shadow.inherit = False
textbox(s, 1.0, 1.55, 11, 0.4, "研究生 · 近期工作汇报", size=15, bold=True, color=RGBColor(0x9D, 0xC3, 0xFF))
textbox(s, 1.0, 2.05, 11.4, 1.1, "超声 / 医学影像 AI 方向：基础能力建设进展", size=38, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
textbox(s, 1.0, 3.25, 11, 0.5, "从「能训练」到「能保存、能复用、能解释」——B1 阶段小结", size=17, color=RGBColor(0xC9, 0xD8, 0xEA))
textbox(s, 1.0, 4.85, 6, 0.4, "汇报人：__________", size=15, color=RGBColor(0xC9, 0xD8, 0xEA))
textbox(s, 1.0, 5.3, 6, 0.4, "2026 年 11 月 3 日", size=15, color=RGBColor(0xC9, 0xD8, 0xEA))
textbox(s, 1.0, 6.55, 11, 0.4,
        "说明：当前为学习与工程练习阶段，Fashion-MNIST 小项目用于打通流程，非医学研究成果。",
        size=11, color=RGBColor(0x8F, 0xA5, 0xBE))

# ---------- 2 一页看懂 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "一页看懂：我在做什么、做到哪", "总览")
textbox(s, 0.7, 1.5, 12, 0.5,
        "主线：为医学影像 / 超声 AI 打地基（深度学习工程能力 + 信号处理 + 论文理解）；具体科研课题尚未确定。",
        size=15, color=GREY)
# 阶段条
stages = [("B1 基础与小实验", "2026-10-29 — 11-18", "当前：收尾", True),
          ("B2 医学数据迁移", "11-19 — 12-16", "计划", False),
          ("B3 工程化", "12-17 — 01-13", "计划", False),
          ("B4 实习准备", "01-14 — 03-14", "计划", False)]
x0, sw, sy = 0.7, 2.95, 2.18
for i, (nm, span, state, cur) in enumerate(stages):
    x = x0 + i * (sw + 0.16)
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(sy), Inches(sw), Inches(0.95))
    r.adjustments[0] = 0.12
    r.fill.solid()
    r.fill.fore_color.rgb = ACCENT if cur else LIGHT
    r.line.color.rgb = ACCENT if cur else LINE
    r.shadow.inherit = False
    textbox(s, x + 0.12, sy + 0.09, sw - 0.2, 0.3, nm, size=13, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF) if cur else NAVY)
    textbox(s, x + 0.12, sy + 0.42, sw - 0.2, 0.26, span, size=10,
            color=RGBColor(0xD6, 0xE6, 0xFF) if cur else GREY)
    textbox(s, x + 0.12, sy + 0.65, sw - 0.2, 0.24, state, size=10, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF) if cur else ACCENT)
# 三件硬东西
card(s, 0.7, 3.4, 3.85, 2.6, "① 训练闭环已打通", [
    ("线性 80.00% / 小 CNN 76.60%", "真实数字，已存盘"),
    ("5 轮 = 395 次参数更新", "配置 / 权重 / 曲线齐全"),
    ("能不看讲义写出五步训练循环", "前向→损失→清零→反向→更新")])
card(s, 4.74, 3.4, 3.85, 2.6, "② 模型能「活过来」", [
    ("新进程 load → eval → predict", "重启 kernel 后独立跑"),
    ("验证集复现 76.60% / 0.6855", "与训练记录一致"),
    ("评价结果不随 batch 变化", "64 / 32 / 1 实测相同")])
card(s, 8.78, 3.4, 3.85, 2.6, "③ 信号能解释", [
    ("FFT 峰位置与幅值换算", "2/N，DC/Nyquist 例外"),
    ("频谱泄漏及如何减轻", "拉长 T → 格点变密"),
    ("与超声成像链的连接", "RF→DAS→包络→B 模式")], accent=GREEN)
textbox(s, 0.7, 6.2, 12, 0.4,
        "当前断点：11/3 已完成 ① ② 与 FFT 泄漏变式；论文第 IV 节进行中，明日继续。",
        size=12.5, bold=True, color=GREY)

# ---------- 3 近期做了什么 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "近期做了哪几件事", "工作内容")
rows = [
    ("CNN 训练闭环", "把只能前向的小 CNN 接进训练循环，跑 1 轮预测 → 5 轮完整训练，真实记录并保存配置、权重、曲线。"),
    ("独立加载评价", "重启 kernel 后重新 load 权重，在验证集上完整评价，复现训练时的 76.60%——证明「保存→加载→复用」成立。"),
    ("同条件对比", "线性 vs 小 CNN：同一数据划分、同一预处理、同一 5 轮预算，做第一张可比的对比表与叠图。"),
    ("信号：FFT 与泄漏", "单/双频真实频谱、幅值换算 2/N、2.0 vs 2.3Hz 泄漏，并把观察时长 T 拉长看泄漏变窄直至消失。"),
    ("论文与前沿", "主论文读到网络结构、训练条件、实验结果；每天 1 篇前沿摘要（扩散重建 / 胎儿 CNS / 乳腺超声）。"),
]
y = 1.42
for i, (h, d) in enumerate(rows):
    nb = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(y), Inches(0.5), Inches(0.9))
    nb.fill.solid(); nb.fill.fore_color.rgb = ACCENT if i < 4 else GREY
    nb.line.fill.background(); nb.shadow.inherit = False
    textbox(s, 0.7, y + 0.2, 0.5, 0.4, f"{i+1}", size=18, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    textbox(s, 1.38, y + 0.06, 11.2, 0.34, h, size=16, bold=True, color=NAVY)
    textbox(s, 1.38, y + 0.4, 11.2, 0.5, d, size=13, color=GREY)
    y += 1.06

# ---------- 4 对比表 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "核心结果①：线性基线 vs 小 CNN", "同条件对比")
table(s,
      ["模型", "参数量", "训练 acc", "验证 acc", "验证 loss", "说明"],
      [["线性 (Flatten+Linear)", "7,850", "80.62%", "80.00%", "0.5666", "基线"],
       ["小 CNN (2 层卷积)", "9,098", "79.74%", "76.60%", "0.6855", "最佳在第 4 轮 78.40%"]],
      left=0.7, top=1.5, width=11.9,
      col_widths=[3.0, 1.35, 1.55, 1.55, 1.55, 2.9], row_h=0.52, body_size=14.5)
# 结论框
bx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(3.1), Inches(11.9), Inches(1.5))
bx.adjustments[0] = 0.06
bx.fill.solid(); bx.fill.fore_color.rgb = RGBColor(0xFF, 0xF6, 0xE5)
bx.line.color.rgb = AMBER; bx.line.width = Pt(1); bx.shadow.inherit = False
textbox(s, 0.9, 3.2, 11.4, 0.35, "诚实结论（照实记录，不做过度推断）", size=14, bold=True, color=AMBER)
textbox(s, 0.9, 3.58, 11.4, 0.95,
        "小 CNN 验证 76.60% < 线性 80.00%；训练 79.74% < 80.62%。这只是一次 5 轮、未调参的观察，"
        "只能说明「本配置下 CNN 没有超过线性」，不能推出「CNN 不如线性」的普遍结论。",
        size=13.5, color=DARK, spacing=1.25)
# 右侧要点
bullets(s, 0.7, 4.95, 11.9, 1.9, [
    ("公平性三要素", "同一验证集成员、同一预处理（仅 ToTensor）、同一 5 轮 / 79 步预算"),
    ("为什么 CNN 前几轮反而低", "卷积网络在本配置下收敛更慢，第 1 轮 62.4% vs 线性 74.0%，属正常现象"),
    ("公平性隐患已修", "验证集来源曾误用测试池，已改回与基线一致的训练池划分"),
], size=14, gap=8)

# ---------- 5 训练曲线 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "核心结果②：两模型训练 / 验证曲线", "叠图")
img = BASE / "learning-lab" / "results" / "day1103" / "linear_vs_cnn_overlay.png"
add_image_fit(s, img, 1.0, 1.42, 11.3, 3.55)
bullets(s, 0.7, 5.15, 11.9, 1.7, [
    ("都在学", "两条曲线的训练损失都持续下降、训练准确率上升"),
    ("CNN 收敛更慢", "线性第 1 轮 val 已 74.0%，CNN 只有 62.4%；第 4 轮 CNN 追到 78.4%"),
    ("轻微过拟合迹象", "CNN 第 5 轮 val loss 回升（0.63→0.69），样本仅 5000，属正常"),
    ("终点一致", "线性 val 80.0% 高于 CNN 76.6%，与对比表一致"),
], size=13.5, gap=2)

# ---------- 6 FFT 泄漏 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "核心结果③：频谱泄漏——怎么看出、怎么减轻", "信号实验")
img = BASE / "learning-lab" / "results" / "day1103" / "leak_vs_T.png"
add_image_fit(s, img, 0.6, 1.38, 6.9, 3.45)
bullets(s, 7.75, 1.5, 5.0, 2.4, [
    ("泄漏是什么", "频率不落在 fs/N 格点上，能量摊到邻近格点，峰变矮、旁瓣出现"),
    ("怎么减轻", "拉长观察时长 T（格点变密 / 加窗同理）"),
    ("实测", "T=1s 峰 0.871；T=10s 峰 1.000"),
], size=13, gap=8)
bx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.75), Inches(4.15), Inches(4.95), Inches(1.45))
bx.adjustments[0] = 0.08
bx.fill.solid(); bx.fill.fore_color.rgb = RGBColor(0xE9, 0xF2, 0xFE)
bx.line.color.rgb = ACCENT; bx.line.width = Pt(1); bx.shadow.inherit = False
textbox(s, 7.93, 4.25, 4.6, 1.25,
        "2.3Hz 在 T=10s 时正好落格点（2.3 = 23×0.1），且为整周期 → 泄漏消失，峰高回到真实幅值 1.00。",
        size=12.5, color=NAVY, spacing=1.2)
textbox(s, 0.6, 5.15, 6.9, 0.9,
        "与超声连接：观察窗越长，频域分辨率越高、越能分开相邻成分（谐波 / 基频），代价是时间分辨率下降——经典时频折中。",
        size=11.5, color=GREY, spacing=1.2)

# ---------- 7 论文 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "论文阅读：读到哪、怎么和我的工作接上", "文献")
card(s, 0.7, 1.45, 6.0, 3.55, "主论文（精读，TUFFC 2021 / arXiv 2008.12750）", [
    ("网络结构", "残差 + 多尺度编解码；下采样用 2×2 步长卷积，跳连用相加"),
    ("训练条件", "Adam，lr 5e-5，batch 2，50 万次迭代 ≈ 33 epochs，3 万对图像"),
    ("损失", "MSLAE（针对高动态范围超声 RF 图设计）"),
    ("实验", "B-mode 下 PSNR / SSIM；单平面波重建质量接近合成孔径金标准"),
    ("与我的差别", "它是「图→图」重建（回归），我是「图→类别」分类；数字不可横比"),
], head_size=13.5, body_size=11.8)
card(s, 6.95, 1.45, 5.9, 3.55, "前沿泛读（每天 1 篇摘要）", [
    ("11/1 扩散模型 3D 超声重建", "看到「重建」不止 CNN 一条路"),
    ("11/2 胎儿 CNS 四类异常（多中心）", "真实医学成绩按「病人」统计，不是按图片"),
    ("11/3 乳腺超声可解释 + 多数据集", "为接下来 BUS-BRA 分类做认知准备"),
    ("共同启发", "准确率之外，要准备「按病例划分」和「可解释」两个口径"),
], accent=GREEN, head_size=13.5, body_size=11.8)
textbox(s, 0.7, 5.25, 12, 0.9,
        "读法：不抠全部公式，按「它解决什么问题 → 输入输出 → 指标与对照 → 局限」四问推进；"
        "每天 1 篇前沿摘要只读标题、摘要、结论，记一句启发。",
        size=12.5, color=GREY, spacing=1.2)

# ---------- 8 诚实边界 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "当前阶段的边界（先讲清楚，避免误读）", "诚实说明")
rows2 = [
    ("数据不是医学数据", "Fashion-MNIST 是练习用的 10 类服饰图；结论只对流程有效，不代表医学性能。"),
    ("对比是单次观察", "一次 5 轮、未做超参搜索；不足以宣称架构优劣，也不足以作为科研结论。"),
    ("未接触临床数据", "课题组数据暂不可用；下一步改用公开乳腺超声数据集 BUS-BRA 练手。"),
    ("没有医学结论", "目前交付的是「工程地基 + 方法理解 + 可复现记录」，不是研究成果、不是论文复现。"),
]
y = 1.5
for h, d in rows2:
    bx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(y), Inches(0.46), Inches(0.46))
    bx.adjustments[0] = 0.25
    bx.fill.solid(); bx.fill.fore_color.rgb = AMBER
    bx.line.fill.background(); bx.shadow.inherit = False
    textbox(s, 0.7, y + 0.06, 0.46, 0.34, "!", size=15, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    textbox(s, 1.35, y - 0.02, 4.4, 0.5, h, size=14.5, bold=True, color=NAVY)
    textbox(s, 5.75, y - 0.02, 6.9, 0.9, d, size=13, color=GREY, spacing=1.15)
    y += 1.22

# ---------- 9 下一步 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "下一步计划（到 11 月中旬）", "计划")
steps = [
    ("11/4 — 11/5", "对比收尾 + 滤波", "补 CNN 混淆矩阵/每类召回；按计划做三点滑动平均滤波（手算 + 代码 + 对比图）。"),
    ("11/5", "准备真实数据", "下载并核对公开乳腺超声数据集 BUS-BRA（1,064 例，BI-RADS 标注），确认目录与标签结构。"),
    ("11/6 — 11/8", "真实超声分类迁移", "复用同一训练闭环换数据：改输入尺寸/通道，做线性 vs CNN，按「病例」而非「图片」划分与统计。"),
    ("11/11", "阶段小结（组会）", "把 B1 的真实结果、图、局限整理成一份小项目报告——只写跑出来的，不预填。"),
]
y = 1.45
for date, h, d in steps:
    textbox(s, 0.7, y + 0.02, 2.3, 0.5, date, size=14, bold=True, color=ACCENT)
    textbox(s, 2.9, y - 0.03, 3.6, 0.5, h, size=15, bold=True, color=NAVY)
    textbox(s, 6.5, y - 0.03, 6.2, 0.95, d, size=12.8, color=GREY, spacing=1.15)
    y += 1.16
ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(6.55), Inches(11.9), Inches(0.02))
ln.fill.solid(); ln.fill.fore_color.rgb = LINE; ln.line.fill.background(); ln.shadow.inherit = False
textbox(s, 0.7, 6.65, 11.9, 0.4,
        "总目标：把「训练→保存→加载→复用」这套骨架，从玩具数据迁到真实超声数据上跑通。",
        size=13, bold=True, color=NAVY)

# ---------- 10 请老师指导 ----------
s = prs.slides.add_slide(BLANK)
frame(s); title(s, "想请老师指导的三个问题", "请教")
qs = [
    ("方向选择", "医学影像/超声 AI 里，第一个真实数据任务更建议先做「分割」还是「重建」？我目前按「分类」入门，是否偏了？"),
    ("数据与合作", "课题组数据短期不可用，先用公开 BUS-BRA 练手是否合适？后续什么阶段可以接触组里的真实数据？"),
    ("深度与节奏", "目前的深度到「能跑通、能解释、能复现」；是继续加深，还是应该更早进入某一篇论文的复现？"),
]
y = 1.55
for i, (h, d) in enumerate(qs, start=1):
    nb = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.75), Inches(y), Inches(0.52), Inches(0.52))
    nb.fill.solid(); nb.fill.fore_color.rgb = ACCENT
    nb.line.fill.background(); nb.shadow.inherit = False
    textbox(s, 0.75, y + 0.08, 0.52, 0.36, f"Q{i}", size=13, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    textbox(s, 1.5, y - 0.05, 4.2, 0.45, h, size=15, bold=True, color=NAVY)
    textbox(s, 1.5, y + 0.42, 11.0, 0.9, d, size=13, color=GREY, spacing=1.2)
    y += 1.6
textbox(s, 0.7, 6.5, 12, 0.5, "谢谢老师，请多指正。", size=17, bold=True, color=NAVY)

out = BASE / "2026-11" / "1103" / "近期工作汇报_2026-11-03.pptx"
prs.save(str(out))
print("saved:", out)
print("slides:", len(prs.slides.__iter__.__self__._sldIdLst))
