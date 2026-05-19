from pathlib import Path
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "assets" / "new_sot_algorithm_architecture.png"

WIDTH, HEIGHT = 2400, 1600
FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def load_font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


F_TITLE = load_font(52, True)
F_SUBTITLE = load_font(27)
F_SECTION = load_font(30, True)
F_BOX = load_font(30, True)
F_BODY = load_font(24)
F_SMALL = load_font(21)

COLORS = {
    "ink": "#202532",
    "muted": "#657084",
    "line": "#748094",
    "blue": "#2563eb",
    "blue_bg": "#eef5ff",
    "green": "#16a34a",
    "green_bg": "#ecfdf3",
    "orange": "#ea580c",
    "orange_bg": "#fff7ed",
    "red": "#dc2626",
    "red_bg": "#fff1f2",
    "gray_bg": "#f7f9fc",
    "gray": "#c7d0dd",
    "slate": "#475569",
}


def wrap_text(text, max_chars):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=max_chars, replace_whitespace=False))
    return lines


def rounded(draw, box, fill, outline="#c7d0dd", width=3, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw, start, end, color=COLORS["line"], width=5, head=18):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    points = []
    for offset in (math.pi * 0.82, -math.pi * 0.82):
        points.append((x2 + head * math.cos(angle + offset), y2 + head * math.sin(angle + offset)))
    draw.polygon([(x2, y2), points[0], points[1]], fill=color)


def draw_polyline(draw, points, color=COLORS["line"], width=5):
    for start, end in zip(points, points[1:-1]):
        draw.line((*start, *end), fill=color, width=width)
    draw_arrow(draw, points[-2], points[-1], color=color, width=width)


def draw_box(draw, box, title, body, fill, outline, accent, max_chars=18):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill, outline=outline, width=3)
    draw.rounded_rectangle((x1, y1, x1 + 14, y2), radius=7, fill=accent)
    draw.text((x1 + 34, y1 + 22), title, font=F_BOX, fill=COLORS["ink"])
    y = y1 + 74
    for line in wrap_text(body, max_chars):
        draw.text((x1 + 34, y), line, font=F_BODY, fill=COLORS["muted"])
        y += 34


def draw_diamond(draw, center, size, text):
    cx, cy = center
    w, h = size
    points = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    draw.polygon(points, fill="#ffffff", outline="#94a3b8")
    draw.line([*points, points[0]], fill="#94a3b8", width=3)
    lines = wrap_text(text, 8)
    heights = [draw.textbbox((0, 0), line, font=F_BOX)[3] for line in lines]
    y = cy - (sum(heights) + (len(lines) - 1) * 8) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=F_BOX)
        draw.text((cx - (bbox[2] - bbox[0]) / 2, y), line, font=F_BOX, fill=COLORS["ink"])
        y += bbox[3] - bbox[1] + 8


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(image)

    draw.text((90, 55), "标准 SOT 单目标验证与重关联算法架构", font=F_TITLE, fill=COLORS["ink"])
    draw.text(
        (92, 125),
        "面向 UAVDT-Benchmark-S；网页 system 模式保持真实掉框语义，benchmark sot 模式每帧输出目标状态",
        font=F_SUBTITLE,
        fill=COLORS["muted"],
    )

    rounded(draw, (70, 200, 655, 1365), "#fbfcff", "#d8dee9", 3, 28)
    rounded(draw, (715, 200, 2328, 1365), "#fbfcff", "#d8dee9", 3, 28)
    draw.text((100, 230), "输入、检测与基础追踪", font=F_SECTION, fill=COLORS["ink"])
    draw.text((750, 230), "评测口径分支与新版 SOT 恢复逻辑", font=F_SECTION, fill=COLORS["ink"])

    left_boxes = [
        ((125, 315, 600, 435), "UAVDT-S 图像序列", "读取官方 SOT 标注\n<seq>_gt.txt", COLORS["blue_bg"], COLORS["blue"]),
        ((125, 505, 600, 625), "首帧 GT 初始化", "用第 1 帧真值框作为\n单目标起点", COLORS["green_bg"], COLORS["green"]),
        ((125, 695, 600, 835), "逐帧 YOLO 检测", "conf / NMS / imgsz\n可选小目标增强", COLORS["orange_bg"], COLORS["orange"]),
        ((125, 925, 600, 1065), "FusionSORTUAV 更新", "检测框输入追踪器\n生成当前轨迹集合", COLORS["gray_bg"], COLORS["line"]),
    ]
    for box, title, body, fill, accent in left_boxes:
        draw_box(draw, box, title, body, fill, "#cad3df", accent)
    for index in range(len(left_boxes) - 1):
        b1 = left_boxes[index][0]
        b2 = left_boxes[index + 1][0]
        draw_arrow(draw, ((b1[0] + b1[2]) / 2, b1[3]), ((b2[0] + b2[2]) / 2, b2[1] - 8), color="#64748b")

    draw_arrow(draw, (600, 995), (750, 995), color="#64748b")
    draw.text((635, 955), "tracks + detections", font=F_SMALL, fill=COLORS["muted"])

    draw_box(
        draw,
        (760, 320, 1175, 500),
        "system 模式",
        "网页诊断口径\nlast_box + IoU 匹配\n失败则输出空框",
        COLORS["red_bg"],
        "#f2b8bd",
        COLORS["red"],
        max_chars=17,
    )
    draw.text((805, 525), "保留原行为：反映真实掉框", font=F_BODY, fill=COLORS["red"])

    draw_box(
        draw,
        (1280, 305, 1715, 455),
        "运动预测",
        "reference_box =\nlast_output_box + velocity",
        COLORS["blue_bg"],
        "#9bbcfb",
        COLORS["blue"],
        max_chars=24,
    )
    draw_box(
        draw,
        (1845, 305, 2245, 455),
        "候选框池",
        "优先检测框\n短时可参考同 ID 轨迹",
        COLORS["green_bg"],
        "#a8ddb8",
        COLORS["green"],
        max_chars=17,
    )
    draw_box(
        draw,
        (1280, 545, 1715, 725),
        "重关联代价",
        "0.45 IoU + 0.40 中心相似\n+ 0.15 尺度相似",
        "#f8fbff",
        "#bfd0e8",
        COLORS["slate"],
        max_chars=25,
    )
    draw_box(
        draw,
        (1845, 545, 2245, 725),
        "动态门控",
        "搜索半径随丢失帧数增大\n尺度相似度 ≥ 0.25",
        COLORS["orange_bg"],
        "#fdba74",
        COLORS["orange"],
        max_chars=19,
    )
    draw_diamond(draw, (1762, 850), (330, 170), "候选匹配\n是否通过？")

    draw_arrow(draw, (1175, 420), (1280, 380), color=COLORS["blue"])
    draw.text((1200, 350), "sot 模式", font=F_BODY, fill=COLORS["blue"])
    draw_arrow(draw, (1715, 380), (1845, 380), color=COLORS["blue"])
    draw_arrow(draw, (1498, 455), (1498, 545), color=COLORS["blue"])
    draw_arrow(draw, (2045, 455), (2045, 545), color=COLORS["blue"])
    draw_polyline(draw, [(1498, 725), (1498, 850), (1597, 850)], color=COLORS["blue"])
    draw_polyline(draw, [(2045, 725), (2045, 850), (1927, 850)], color=COLORS["blue"])

    draw_polyline(draw, [(1660, 935), (1490, 935), (1490, 975)], color=COLORS["green"])
    draw.text((1530, 907), "是", font=F_BODY, fill=COLORS["green"])
    draw_polyline(draw, [(1875, 935), (2050, 935), (2050, 975)], color=COLORS["red"])
    draw.text((1980, 907), "否", font=F_BODY, fill=COLORS["red"])

    draw_box(
        draw,
        (1265, 975, 1715, 1145),
        "观测成功",
        "更新 observed_box / velocity / track_id\nmissing = 0",
        COLORS["green_bg"],
        "#86efac",
        COLORS["green"],
        max_chars=28,
    )
    draw_box(
        draw,
        (1845, 975, 2245, 1145),
        "短时预测输出",
        "输出 reference_box\nmissing_observations + 1",
        COLORS["red_bg"],
        "#fecdd3",
        COLORS["red"],
        max_chars=23,
    )

    note = (
        "关键变化：不是改网页追踪语义，而是在离线 UAVDT-S 验证中新增标准 SOT 口径；\n"
        "丢失后用预测框续接，并用中心距离与尺度约束恢复重新出现的检测目标。"
    )
    y = 1210
    for line in note.split("\n"):
        draw.text((760, y), line, font=F_BODY, fill=COLORS["muted"])
        y += 38

    rounded(draw, (130, 1410, 2270, 1535), "#f8fafc", "#cbd5e1", 3, 24)
    draw.text((165, 1440), "输出与论文指标", font=F_SECTION, fill=COLORS["ink"])
    output_items = [
        (520, "每帧预测框 TXT"),
        (875, "per_frame.csv 诊断"),
        (1240, "Precision / Success Plot"),
        (1665, "Precision@20、AUC、Success@0.5"),
        (2080, "最长丢失、检测召回、重关联率"),
    ]
    for x, label in output_items:
        draw.ellipse((x - 13, 1484 - 13, x + 13, 1484 + 13), fill=COLORS["blue"])
        bbox = draw.textbbox((0, 0), label, font=F_SMALL)
        draw.text((x - (bbox[2] - bbox[0]) / 2, 1506), label, font=F_SMALL, fill=COLORS["muted"])

    draw_polyline(draw, [(950, 500), (950, 1375), (520, 1410)], color="#94a3b8", width=4)
    draw_polyline(draw, [(1490, 1145), (1490, 1375), (1240, 1410)], color="#94a3b8", width=4)
    draw_polyline(draw, [(2050, 1145), (2050, 1375), (1665, 1410)], color="#94a3b8", width=4)

    image.save(OUTPUT_PATH, quality=95)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
