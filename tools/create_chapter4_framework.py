from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "chapter4_algorithm_framework_portrait.png"


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


W, H = 1600, 2350
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

title_font = font(48, True)
section_font = font(30, True)
node_font = font(25)
small_font = font(21)

ink = (24, 35, 48)
line = (58, 72, 88)
blue = (229, 240, 251)
blue_edge = (88, 125, 160)
green = (232, 247, 238)
green_edge = (88, 146, 105)
orange = (253, 242, 228)
orange_edge = (180, 125, 61)
gray = (246, 248, 251)


def text_size(text, fnt):
    box = d.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def lines_for(text, max_chars):
    lines = []
    for part in text.split("\n"):
        lines.extend(wrap(part, max_chars, break_long_words=False) or [""])
    return lines


def node(x, y, w, h, text, fill="white", edge=line, max_chars=13, fnt=node_font):
    d.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=fill, outline=edge, width=3)
    lines = lines_for(text, max_chars)
    line_h = fnt.size + 8
    total_h = line_h * len(lines) - 8
    yy = y + (h - total_h) / 2
    for ln in lines:
        tw, _ = text_size(ln, fnt)
        d.text((x + (w - tw) / 2, yy), ln, fill=ink, font=fnt)
        yy += line_h


def section(x, y, w, h, title, fill):
    d.rounded_rectangle((x, y, x + w, y + h), radius=24, fill=fill, outline=line, width=3)
    d.text((x + 28, y + 22), title, fill=ink, font=section_font)


def arrow(x1, y1, x2, y2, width=4):
    import math

    d.line((x1, y1, x2, y2), fill=line, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 17
    pts = [
        (x2, y2),
        (x2 - size * math.cos(ang - 0.48), y2 - size * math.sin(ang - 0.48)),
        (x2 - size * math.cos(ang + 0.48), y2 - size * math.sin(ang + 0.48)),
    ]
    d.polygon(pts, fill=line)


def h_arrow(x1, y, x2):
    arrow(x1, y, x2, y)


def v_arrow(x, y1, y2):
    arrow(x, y1, x, y2)


def title_center(text, y, fnt, color=ink):
    tw, _ = text_size(text, fnt)
    d.text(((W - tw) / 2, y), text, fill=color, font=fnt)


title_center("算法整体框架", 45, title_font)
d.line((160, 125, W - 160, 125), fill=(135, 154, 174), width=2)

# 1. Interaction stage.
section(70, 170, W - 140, 430, "一、交互式初始化与总体流程", blue)
top_w, top_h, top_gap = 210, 82, 25
top_x0, top_y = 95, 305
top_items = [
    "输入无人机视频",
    "设置检测参数",
    "选择关键帧",
    "关键帧检测与分割",
    "候选目标编号",
    "人工选定追踪目标",
]
for i, t in enumerate(top_items):
    x = top_x0 + i * (top_w + top_gap)
    node(x, top_y, top_w, top_h, t, edge=blue_edge, max_chars=8)
    if i:
        h_arrow(x - top_gap + 7, top_y + top_h / 2, x - 7)
node(1265, 455, 230, top_h, "输出初始目标框", edge=blue_edge, max_chars=8)
v_arrow(1265 + 115, top_y + top_h, 455)

# 2. Detection module.
section(70, 675, 690, 820, "二、目标检测与实例分割算法", green)
d.text((105, 728), "代码入口：detect_and_segment_on_frame() / _infer_frame()", fill=(65, 95, 73), font=small_font)
det_items = [
    "权重检查与模型加载",
    "图像预处理与尺寸调整",
    "YOLOv8 前向推理",
    "解析边界框、类别、置信度和掩码",
    "小目标增强推理",
    "NMS融合与车辆筛选",
    "输出结构化候选目标集",
]
box_x, box_w, box_h = 150, 530, 74
start_y, step = 790, 95
for i, t in enumerate(det_items):
    y = start_y + i * step
    node(box_x, y, box_w, box_h, t, edge=green_edge, max_chars=17)
    if i:
        v_arrow(box_x + box_w / 2, y - (step - box_h), y - 8)

# 3. Tracking module.
section(840, 675, 690, 820, "三、检测驱动的单目标追踪算法", orange)
d.text((875, 728), "代码入口：track_selected_object() 调用 FusionSORTUAV", fill=(105, 75, 40), font=small_font)
trk_items = [
    "重新打开视频并初始化追踪器",
    "逐帧执行目标检测",
    "转换追踪器输入格式",
    "卡尔曼预测与轨迹更新",
    "IoU匹配与数据关联",
    "关键帧轨迹ID绑定",
    "指定ID轨迹渲染并写入视频",
]
box_x2 = 920
for i, t in enumerate(trk_items):
    y = start_y + i * step
    node(box_x2, y, box_w, box_h, t, edge=orange_edge, max_chars=17)
    if i:
        v_arrow(box_x2 + box_w / 2, y - (step - box_h), y - 8)

# Cross-stage arrows.
v_arrow(415, 600, 675)
v_arrow(1395, 517, 675)
v_arrow(415, 1495, 1605)
v_arrow(1185, 1495, 1605)

# 4. Output.
section(150, 1605, W - 300, 350, "四、结果输出与可视化反馈", gray)
out_w, out_h = 320, 78
out_x = [255, 640, 1025]
out_y = 1740
out_items = ["绑定目标ID", "叠加目标框、ID和可选掩码", "生成单目标追踪视频"]
for i, t in enumerate(out_items):
    node(out_x[i], out_y, out_w, out_h, t, edge=line, max_chars=13)
    if i:
        h_arrow(out_x[i - 1] + out_w + 12, out_y + out_h / 2, out_x[i] - 12)

note = (
    "说明：检测分割模块由 YOLOv8 输出边界框、类别、置信度和实例掩码，并结合小目标增强、NMS 融合与车辆筛选形成候选目标集；"
    "追踪模块以 FusionSORTUAV 为核心，通过卡尔曼预测、IoU 数据关联和轨迹状态维护实现在线单目标跟踪。"
)
node(150, 2055, W - 300, 180, note, fill=(250, 250, 250), edge=(168, 180, 194), max_chars=44, fnt=small_font)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, quality=95)
print(OUT)
