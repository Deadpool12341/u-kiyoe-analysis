"""
改进的多尺度多角度模板匹配 - 优化版
主要改进：
1. 更细粒度的尺度(0.5-1.5, 步长0.1)和角度(-90°~90°, 步长5°)搜索
2. 自适应阈值：根据尺度自动调整匹配阈值
3. 改进的NMS：考虑角度差异，避免过度抑制不同方向的相似形状
4. 可视化增强：保存原始检测、按置信度着色、输出详细信息文件
5. 可调参数：通过修改CONFIG字典轻松调整算法参数
"""

import cv2
import numpy as np
import os

# Verify OpenCV installation
try:
    if not hasattr(cv2, 'imread'):
        raise ImportError(
            "OpenCV installation is broken. 'cv2.imread' not found.\n"
            "Please reinstall OpenCV by running:\n"
            "  pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y\n"
            "  pip install opencv-python\n"
        )
except Exception as e:
    print(f"ERROR: {e}")
    raise

# ====== 固定输入路径 ======
IMG_PATH = r"C:\Users\wang6\Desktop\research\code\claw analysis\input.png"
TEMPLATE_PATH = r"C:\Users\wang6\Desktop\research\code\claw analysis\template.png"
BASE_DIR = r"C:\Users\wang6\Desktop\research\code\claw analysis"
# ==========================

# ====== 可调参数配置 ======
CONFIG = {
    # 尺度范围：[最小尺度, 最大尺度, 步长]
    "scale_range": (0.5, 1.6, 0.1),

    # 角度范围：[最小角度, 最大角度, 步长]
    "angle_range": (-90, 91, 5),

    # 阈值设置
    "base_threshold": 0.32,  # 基础匹配阈值（较小尺度）
    "extreme_threshold": 0.40,  # 极端尺度的阈值
    "min_confidence": 0.35,  # 最终输出的最低置信度

    # NMS参数
    "nms_iou_thresh": 0.5,  # IoU阈值
    "nms_angle_diff": 15,  # 角度差异阈值（度）

    # 边缘检测参数
    "big_canny": (40, 140),  # 大图Canny参数
    "tmpl_canny": (30, 120),  # 模板Canny参数
}
# ==========================


def get_unique_outdir(base_dir: str, base_name: str = "pattern matching") -> str:
    outdir = os.path.join(base_dir, base_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        return outdir
    i = 1
    while True:
        cand = os.path.join(base_dir, f"{base_name}{i}")
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=True)
            return cand
        i += 1


def rotate_keep_bounds(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def nms(detections, iou_thresh=0.5, angle_diff_thresh=15):
    # detections: (score, x, y, w, h, angle, scale)
    # 改进的NMS：考虑角度差异，避免过度抑制
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    picked = []
    for det in detections:
        score, x, y, w, h, ang, scl = det
        keep = True
        for p in picked:
            _, px, py, pw, ph, pang, _ = p

            # 计算角度差异（考虑周期性）
            angle_diff = abs(ang - pang)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # 如果角度差异很大，可能是不同方向的爪子，降低IoU要求
            if angle_diff > angle_diff_thresh:
                effective_iou_thresh = min(iou_thresh * 1.3, 0.8)
            else:
                effective_iou_thresh = iou_thresh

            # 计算IoU
            xx1 = max(x, px)
            yy1 = max(y, py)
            xx2 = min(x + w, px + pw)
            yy2 = min(y + h, py + ph)
            ww = max(0, xx2 - xx1)
            hh = max(0, yy2 - yy1)
            inter = ww * hh
            union = w * h + pw * ph - inter
            iou = inter / union if union > 0 else 0

            if iou > effective_iou_thresh:
                keep = False
                break
        if keep:
            picked.append(det)
    return picked


def main():
    out_dir = get_unique_outdir(BASE_DIR, "pattern matching")
    print(">>> output to:", out_dir)

    # 1) 大图
    big = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if big is None:
        raise FileNotFoundError(IMG_PATH)
    big_blur = cv2.GaussianBlur(big, (5, 5), 0)
    big_edge = cv2.Canny(big_blur, *CONFIG["big_canny"])
    cv2.imwrite(os.path.join(out_dir, "debug_big_edge.png"), big_edge)

    # 2) 模板边缘
    tmpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise FileNotFoundError(TEMPLATE_PATH)
    tmpl_blur = cv2.GaussianBlur(tmpl, (3, 3), 0)
    tmpl_edge = cv2.Canny(tmpl_blur, *CONFIG["tmpl_canny"])
    cv2.imwrite(os.path.join(out_dir, "debug_template_edge.png"), tmpl_edge)

    # 3) 多尺度 + 多角度匹配 - 改进版
    # 从CONFIG读取参数
    scales = np.arange(*CONFIG["scale_range"]).tolist()
    angles = list(range(*CONFIG["angle_range"]))
    method = cv2.TM_CCOEFF_NORMED

    print(f">>> Searching {len(scales)} scales × {len(angles)} angles = {len(scales)*len(angles)} combinations")

    # 使用自适应阈值 - 不同尺度可能需要不同阈值
    def adaptive_threshold(scale):
        # 极小或极大的缩放可能匹配质量下降，提高阈值
        if scale < 0.7 or scale > 1.4:
            return CONFIG["extreme_threshold"]
        return CONFIG["base_threshold"]

    detections = []
    total_iterations = len(scales) * len(angles)
    iteration = 0

    for scl in scales:
        # 缩放模板
        th, tw = tmpl_edge.shape[:2]
        new_w = int(tw * scl)
        new_h = int(th * scl)
        if new_w < 5 or new_h < 5:
            continue
        scaled_tmpl = cv2.resize(tmpl_edge, (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)

        for ang in angles:
            iteration += 1
            if iteration % 50 == 0:
                print(f"  Progress: {iteration}/{total_iterations}")

            rot_tmpl = rotate_keep_bounds(scaled_tmpl, ang)
            rh, rw = rot_tmpl.shape[:2]
            if rh < 5 or rw < 5:
                continue

            res = cv2.matchTemplate(big_edge, rot_tmpl, method)

            # 只保存关键角度的heatmap，避免生成太多文件
            if ang % 30 == 0:
                heat = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(out_dir, f"heat_s{scl:.2f}_a{ang}.png"), heat)

            # 使用自适应阈值
            threshold = adaptive_threshold(scl)
            loc = np.where(res >= threshold)
            for py, px in zip(*loc):
                score = float(res[py, px])
                detections.append((score, px, py, rw, rh, ang, scl))

    print(f">>> raw detections: {len(detections)}")
    if not detections:
        print("!!! no detections, try lowering adaptive_threshold values")
        return

    # 可视化所有原始检测（NMS前）- 帮助调试
    vis_raw = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    for score, x, y, w, h, ang, scl in detections[:500]:  # 最多显示500个
        cv2.rectangle(vis_raw, (x, y), (x + w, y + h), (0, 255, 255), 1)
    cv2.imwrite(os.path.join(out_dir, "raw_detections.png"), vis_raw)
    print(">>> saved raw detections visualization")

    # 4) 改进的NMS - 降低IoU阈值，更好地保留不同的爪子
    picked = nms(detections,
                 iou_thresh=CONFIG["nms_iou_thresh"],
                 angle_diff_thresh=CONFIG["nms_angle_diff"])
    print(f">>> after NMS: {len(picked)}")

    # 可选：按置信度过滤（如果结果太多）
    picked = [d for d in picked if d[0] >= CONFIG["min_confidence"]]
    print(f">>> after confidence filter (>={CONFIG['min_confidence']}): {len(picked)}")

    # 5) 画最终结果 - 使用不同颜色区分置信度
    vis = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    for i, (score, x, y, w, h, ang, scl) in enumerate(picked):
        # 根据置信度选择颜色：高置信度=红色，中等=橙色，低=黄色
        if score >= 0.50:
            color = (0, 0, 255)  # 红色 - 高置信度
        elif score >= 0.40:
            color = (0, 165, 255)  # 橙色 - 中等置信度
        else:
            color = (0, 255, 255)  # 黄色 - 较低置信度

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # 标注编号和信息
        label = f"#{i+1} {score:.2f}"
        cv2.putText(vis, label, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)

    out_path = os.path.join(out_dir, "matches.png")
    cv2.imwrite(out_path, vis)
    print(">>> saved:", out_path)

    # 保存检测详情到文本文件
    detail_path = os.path.join(out_dir, "detections.txt")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(f"Total detections: {len(picked)}\n")
        f.write("=" * 60 + "\n")
        for i, (score, x, y, w, h, ang, scl) in enumerate(picked):
            f.write(f"#{i+1}: score={score:.3f}, pos=({x},{y}), "
                    f"size=({w}x{h}), angle={ang}°, scale={scl:.2f}\n")
    print(f">>> saved detection details: {detail_path}")
    print(">>> done")


if __name__ == "__main__":
    main()
