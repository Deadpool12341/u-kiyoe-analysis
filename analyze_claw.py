# -*- coding: utf-8 -*-
"""
Analyze Hokusai-style 'claw' tips from an image.

NEW APPROACH - Centerline-based analysis:
- 从爪形提取中心线（medial axis）作为分析基础
- 识别4个关节点：Joint 1 (根部) + Joint 2&3 (中心线上曲率最大的两点) + Joint 4 (尖端)
- 可视化：roiXX_joints.png（绿色小圆圈标记关节，红色直线连接关节，黄色显示中心线）
- 输出：4个关节坐标、边长、对角线长度、角度、中心线曲率等指标
- 交互 ROI：兼容 matplotlib 3.10 的 RectangleSelector
- 输出目录自动递增：out, out1, out2, ...
- 依赖：numpy scipy opencv-python scikit-image matplotlib
"""

import os
import argparse
import csv
import math
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from skimage.morphology import medial_axis
from skimage.graph import route_through_array
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks, savgol_filter

# -----------------------------
# Utils
# -----------------------------
def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def mkdir_p(path: str):
    os.makedirs(path, exist_ok=True)

def next_available_outdir(parent_dir: str, base_name: str = "out") -> str:
    """返回 parent_dir 下第一个可用目录名：out, out1, out2, ..."""
    cand = os.path.join(parent_dir, base_name)
    if not os.path.exists(cand):
        return cand
    i = 1
    while True:
        cand_i = os.path.join(parent_dir, f"{base_name}{i}")
        if not os.path.exists(cand_i):
            return cand_i
        i += 1

def imread_rgb(path: str):
    """
    支持中文/PNG/RGBA/灰度路径读取，返回 RGB 和可选 alpha
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot read image: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = None
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        alpha = img[:, :, 3]
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = None
    return rgb, alpha

# -----------------------------
# 1) Binarization（黑白图更稳）
# -----------------------------
def binarize_roi(img_rgb, img_alpha=None, black_on_white=True):
    """
    更稳健的二值化：自动在两种极性之间选择更“像带状”的掩模。
    - 去边框粘连：把ROI四周2px清零，避免整块被选中
    - Otsu + 形态学 + 最大连通域 + 填孔
    - 评分选择：骨架长度/面积，端点数>=2，加上触边惩罚
    """
    import numpy as np
    import cv2
    from skimage.morphology import medial_axis

    H, W = img_rgb.shape[:2]

    def _prep(gray, invert=False):
        # Otsu
        _th_type = cv2.THRESH_BINARY
        _, th = cv2.threshold(gray, 0, 255, _th_type + cv2.THRESH_OTSU)
        if invert:
            th = cv2.bitwise_not(th)

        # 去掉边框粘连（2px）
        th[:2, :] = 0; th[-2:, :] = 0; th[:, :2] = 0; th[:, -2:] = 0

        # 形态学
        k3 = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k3, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k3, iterations=1)

        # 只保留最大连通域
        num, labels = cv2.connectedComponents(th)
        if num > 1:
            areas = [(labels == i).sum() for i in range(1, num)]
            i_best = 1 + int(np.argmax(areas))
            th = np.where(labels == i_best, 255, 0).astype('uint8')

        # 填孔
        th_inv = cv2.bitwise_not(th)
        flood = np.zeros((H + 2, W + 2), np.uint8)
        fill = th.copy()
        cv2.floodFill(fill, flood, (0, 0), 255)
        holes = cv2.bitwise_not(fill) & th_inv
        th = cv2.bitwise_or(th, holes)

        # 评分：骨架长度/面积 & 端点数 & 触边惩罚
        mask = (th > 0).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            return th, -1e9  # 极差
        skel = medial_axis(mask)
        sk_len = int(skel.sum())
        # 端点数（8邻域）
        kk = np.ones((3, 3), np.uint8)
        nb = cv2.filter2D(skel.astype(np.uint8), -1, kk, borderType=cv2.BORDER_CONSTANT)
        endpoints = int(np.sum((skel == 1) & (nb == 2)))

        score = (sk_len / (area + 1e-3)) + 0.2 * (endpoints >= 2)
        # 触边惩罚：若骨架贴到四边，扣分
        touch = (skel[0, :].any() or skel[-1, :].any() or skel[:, 0].any() or skel[:, -1].any())
        if touch:
            score -= 0.5
        return th, float(score)

    # 若有 alpha，优先用 alpha 当前景
    if img_alpha is not None:
        th = (img_alpha > 10).astype('uint8') * 255
        th[:2, :] = 0; th[-2:, :] = 0; th[:, :2] = 0; th[:, -2:] = 0
        return th

    # 灰度
    bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 两种极性都试，选分数高的
    th1, sc1 = _prep(gray, invert=False)
    th2, sc2 = _prep(gray, invert=True)
    th = th1 if sc1 >= sc2 else th2
    return th


# -----------------------------
# 2) Edge curve extraction for claw analysis
# -----------------------------
def extract_edges_and_centerline(roi_rgb: np.ndarray, mask_u8: np.ndarray, resample_L: int = 256):
    """
    从黑色像素直接提取两条边缘曲线，然后计算中心线
    1. 找到所有黑色像素（这些是实际画的边缘）
    2. 使用聚类分成两条边
    3. 计算中心线作为两条边对应点的中点

    返回：edge1_xy, edge2_xy, centerline_xy, tip_point, root_point, black_pixels
    """
    # 转换为灰度图
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    # 提取黑色像素（阈值化）
    # 使用更高的阈值捕获灰色边缘像素（抗锯齿效果）
    _, black_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 轻度形态学闭运算，仅连接小间隙，不增厚边缘
    kernel = np.ones((2, 2), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 找到所有黑色像素点
    ys, xs = np.where(black_mask > 0)
    if len(xs) < 20:
        raise ValueError("Not enough black pixels found")

    black_pixels = np.stack([xs, ys], axis=1)  # (N, 2) in (x, y)

    # 使用连通组件分析分成两条边
    # 对黑色mask进行连通组件标记
    num_labels, labels_img = cv2.connectedComponents(black_mask)

    if num_labels < 3:  # background + 2 edges
        raise ValueError(f"Could not find 2 distinct edges, found {num_labels-1} components")

    # 找到两个最大的组件（排除背景 label=0）
    component_sizes = []
    for label in range(1, num_labels):
        size = np.sum(labels_img == label)
        component_sizes.append((label, size))

    component_sizes.sort(key=lambda x: x[1], reverse=True)

    if len(component_sizes) < 2:
        raise ValueError("Could not find 2 distinct edge components")

    label1, label2 = component_sizes[0][0], component_sizes[1][0]

    # 提取每条边的像素
    edge1_mask = (labels_img == label1)
    edge2_mask = (labels_img == label2)

    ys1, xs1 = np.where(edge1_mask)
    ys2, xs2 = np.where(edge2_mask)

    edge1_points = np.stack([xs1, ys1], axis=1)
    edge2_points = np.stack([xs2, ys2], axis=1)

    # 对每条边进行排序（沿着曲线方向）
    def sort_curve_points(points):
        """使用最近邻方法对点进行排序"""
        if len(points) < 2:
            return points

        # 从一个端点开始
        from scipy.spatial.distance import pdist, squareform
        if len(points) > 3000:
            # 降采样
            indices = np.random.choice(len(points), 3000, replace=False)
            sample_pts = points[indices]
        else:
            sample_pts = points
            indices = np.arange(len(points))

        D = squareform(pdist(sample_pts, 'euclidean'))
        i, j = np.unravel_index(np.argmax(D), D.shape)

        # 从 i 开始，贪心选择最近的未访问点
        if len(points) <= 3000:
            sorted_pts = [points[i]]
            remaining = set(range(len(points))) - {i}
        else:
            start_idx = indices[i]
            sorted_pts = [points[start_idx]]
            remaining = set(range(len(points))) - {start_idx}

        current = sorted_pts[0]
        while remaining:
            # 找最近的点
            remaining_pts = points[list(remaining)]
            dists = np.linalg.norm(remaining_pts - current, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_global = list(remaining)[nearest_idx]

            sorted_pts.append(points[nearest_global])
            remaining.remove(nearest_global)
            current = points[nearest_global]

        return np.array(sorted_pts)

    edge1_sorted = sort_curve_points(edge1_points)
    edge2_sorted = sort_curve_points(edge2_points)

    # 平滑并重采样
    def smooth_and_resample(pts, L):
        if len(pts) < 4:
            return pts
        # 去重
        unique_pts = []
        for p in pts:
            if not unique_pts or np.linalg.norm(p - unique_pts[-1]) > 0.5:
                unique_pts.append(p)
        pts = np.array(unique_pts)

        if len(pts) < 4:
            return pts

        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts)*0.5, per=False)
        u = np.linspace(0, 1, L)
        x, y = splev(u, tck)
        return np.stack([x, y], axis=1)

    edge1_smooth = smooth_and_resample(edge1_sorted, resample_L)
    edge2_smooth = smooth_and_resample(edge2_sorted, resample_L)

    # 确保两条边方向一致（从同一端开始）
    # 检查起点距离和终点距离
    dist_start_start = np.linalg.norm(edge1_smooth[0] - edge2_smooth[0])
    dist_start_end = np.linalg.norm(edge1_smooth[0] - edge2_smooth[-1])

    if dist_start_end < dist_start_start:
        edge2_smooth = edge2_smooth[::-1]

    # 计算中心线：两条边对应点的中点
    centerline_xy = (edge1_smooth + edge2_smooth) / 2.0

    # 确定根和尖：计算每个位置的"宽度"（两条边之间的距离）
    widths = np.linalg.norm(edge1_smooth - edge2_smooth, axis=1)

    # 尖端 = 宽度最小的地方
    tip_idx = np.argmin(widths)
    # 根部 = 另一端
    root_idx = 0 if tip_idx > len(widths) // 2 else len(widths) - 1

    tip_point = centerline_xy[tip_idx]
    root_point = centerline_xy[root_idx]

    return edge1_smooth, edge2_smooth, centerline_xy, tip_point, root_point, black_mask

def compute_curvature(xy: np.ndarray, sg_window: int = 15, sg_poly: int = 3):
    """
    计算曲线的曲率
    xy: (N, 2) array of (x, y) coordinates
    返回：kappa (N,) 曲率数组
    """
    # 计算一阶和二阶导数
    dx = savgol_filter(xy[:, 0], ensure_odd(sg_window), sg_poly, deriv=1)
    dy = savgol_filter(xy[:, 1], ensure_odd(sg_window), sg_poly, deriv=1)
    ddx = savgol_filter(xy[:, 0], ensure_odd(sg_window), sg_poly, deriv=2)
    ddy = savgol_filter(xy[:, 1], ensure_odd(sg_window), sg_poly, deriv=2)

    # 曲率公式：κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numer = np.abs(dx * ddy - dy * ddx)
    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-8)
    kappa = numer / denom

    return kappa

def find_four_joints(centerline_xy, root_point, tip_point):
    """
    找到爪的4个关节点：
    - Joint 1: 根部端点 (root)
    - Joint 2 & 3: 中心线上曲率最大的两个点
    - Joint 4: 尖端点 (tip)

    返回：4个关节点的坐标 (4, 2), 中心线曲率
    """
    # 计算中心线的曲率
    kappa = compute_curvature(centerline_xy, sg_window=21, sg_poly=3)

    # 找到根和尖在中心线上的索引
    root_dists = np.linalg.norm(centerline_xy - root_point, axis=1)
    tip_dists = np.linalg.norm(centerline_xy - tip_point, axis=1)
    root_idx = np.argmin(root_dists)
    tip_idx = np.argmin(tip_dists)

    # 确保 root_idx < tip_idx
    if root_idx > tip_idx:
        root_idx, tip_idx = tip_idx, root_idx

    # 忽略靠近端点的区域（避免端点效应）
    margin = max(10, len(kappa) // 10)
    kappa_masked = kappa.copy()
    kappa_masked[:margin] = 0
    kappa_masked[-margin:] = 0

    # 找到曲率最大的两个点
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(kappa_masked, distance=20, prominence=0.01)

    if len(peaks) < 2:
        # 如果找不到两个峰，就直接取最大的两个曲率点
        sorted_indices = np.argsort(kappa_masked)[::-1]
        # 过滤掉太近的点
        selected = [sorted_indices[0]]
        for idx in sorted_indices[1:]:
            if all(abs(idx - s) > 20 for s in selected):
                selected.append(idx)
                if len(selected) >= 2:
                    break
        peaks = np.array(sorted(selected))

    # 取曲率最大的两个峰
    if len(peaks) > 2:
        peak_values = kappa[peaks]
        top2_indices = np.argsort(peak_values)[::-1][:2]
        peaks = np.sort(peaks[top2_indices])

    if len(peaks) >= 2:
        joint2_idx, joint3_idx = peaks[0], peaks[1]
    elif len(peaks) == 1:
        joint2_idx = peaks[0]
        joint3_idx = np.argmax(kappa_masked)
    else:
        # 兜底：均分中间区域
        mid = len(centerline_xy) // 2
        joint2_idx = mid - len(centerline_xy) // 6
        joint3_idx = mid + len(centerline_xy) // 6

    # 4个关节点：按照沿中心线的顺序
    joints = np.array([
        root_point,                    # Joint 1: 根部
        centerline_xy[joint2_idx],     # Joint 2: 第一个高曲率点
        centerline_xy[joint3_idx],     # Joint 3: 第二个高曲率点
        tip_point                       # Joint 4: 尖端
    ])

    return joints, kappa

# -----------------------------
# 3) Curvature & arclength（可调平滑）
# -----------------------------
def curvature_on_centerline(xy: np.ndarray, smooth_s: float = 0.2, sg_window: int = 9, sg_poly: int = 2):
    """
    样条拟合后计算 κ(s) 与弧长 s；增加 sg_window/sg_poly 可调，默认平滑较轻以保留尖峰。
    返回：xy_s, s, kappa_s
    """
    tck, _ = splprep([xy[:, 0], xy[:, 1]], s=smooth_s)
    u = np.linspace(0.0, 1.0, len(xy))
    x, y = splev(u, tck, der=0)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-8)
    kappa = np.abs(dx * ddy - dy * ddx) / denom

    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.hstack([[0.0], np.cumsum(seg)])

    sg_window = ensure_odd(max(5, min(sg_window, len(kappa) - (1 - len(kappa) % 2))))
    kappa_s = savgol_filter(kappa, sg_window, sg_poly, mode='interp')
    xy_s = np.stack([x, y], axis=1)
    return xy_s, s, kappa_s

# -----------------------------
# 4) 关节选择（内部区间 κ 全局最大）
# -----------------------------
def find_joints_from_kappa(kappa_s: np.ndarray,
                           min_distance: int = 15,
                           prominence: float = 0.02,
                           max_joints: int = 3):
    peaks, info = find_peaks(kappa_s, distance=min_distance, prominence=prominence)
    if len(peaks) > max_joints:
        prom = info['prominences']
        keep = np.argsort(prom)[::-1][:max_joints]
        peaks = np.sort(peaks[keep])

    idx = [0] + list(map(int, peaks)) + [len(kappa_s) - 1]
    segments = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
    return peaks.astype(int), segments

def pick_three_joint_indices(kappa_s, peaks, L, margin_frac=0.05):
    """
    返回 (j1, j2, j3)：
      j1 = 0（起点/根）
      j3 = L-1（终点/尖）
      j2 = κ(s) 在“内部区间”的全局最大点（不依赖 find_peaks；两端各去 margin）
    """
    j1, j3 = 0, L - 1
    m = max(1, int(L * margin_frac))
    if L - 2 * m <= 1:
        j2 = int(np.argmax(kappa_s))
    else:
        interior = slice(m, L - m)
        j2 = int(np.argmax(kappa_s[interior]) + m)
    return j1, j2, j3

# -----------------------------
# 5) 直线段度量（长度/比值/夹角）
# -----------------------------
def straight_segment_metrics(xy, j1, j2, j3):
    """
    用两条直线段 1→2、2→3 计算：
      len12, len23（像素）
      ratio12, ratio23（以 len12+len23 归一）
      angle_deg（在 j2 处的夹角，[-180,180]，逆时针为正）
    并返回三个关节点坐标
    """
    p1 = xy[j1]; p2 = xy[j2]; p3 = xy[j3]
    v12 = p2 - p1; v23 = p3 - p2
    len12 = float(np.hypot(*(v12)))
    len23 = float(np.hypot(*(v23)))
    total = max(len12 + len23, 1e-8)
    ratio12 = len12 / total
    ratio23 = len23 / total
    cross = v12[0] * v23[1] - v12[1] * v23[0]
    dot = v12[0] * v23[0] + v12[1] * v23[1]
    angle_deg = float(math.degrees(math.atan2(cross, dot)))
    return (len12, len23, ratio12, ratio23, angle_deg, p1, p2, p3)

# -----------------------------
# 6) 可视化
# -----------------------------
def visualize_claw_edges(roi_rgb, mask, edge1_xy, edge2_xy, centerline_xy, joints, kappa,
                         outdir, tag, overlay_scale=2, edges=None):
    """
    可视化爪形的边缘和4个关节点
    - 画中心线（黄色）
    - 画4个关节点（绿色小圆圈）
    - 用红色直线连接关节点（不形成闭合）
    """
    mkdir_p(outdir)

    # 创建叠加图像
    overlay = roi_rgb.copy()

    # 画中心线（黄色）
    centerline_pts = centerline_xy.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [centerline_pts], isClosed=False, color=(255, 255, 0),
                  thickness=2, lineType=cv2.LINE_AA)

    # 连接关节点的直线（红色，不连接最后一个和第一个）
    # 只画 3 条线：joint0->joint1, joint1->joint2, joint2->joint3
    for i in range(3):
        p1 = joints[i]
        p2 = joints[i + 1]
        pt1 = tuple(np.round(p1).astype(int))
        pt2 = tuple(np.round(p2).astype(int))
        cv2.line(overlay, pt1, pt2, (255, 0, 0), 2, lineType=cv2.LINE_AA)

    # 画4个关节点（绿色，较小的圆圈）
    for joint in joints:
        x, y = int(joint[0]), int(joint[1])
        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 4, (0, 200, 0), 1, lineType=cv2.LINE_AA)  # 外圈

    # 保存叠加图像
    cv2.imwrite(os.path.join(outdir, f"{tag}_joints.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 保存放大版本
    if overlay_scale and overlay_scale > 1:
        H, W = overlay.shape[:2]
        big = cv2.resize(overlay, (W * overlay_scale, H * overlay_scale),
                         interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(outdir, f"{tag}_joints_x{overlay_scale}.png"),
                    cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

    # 保存mask
    cv2.imwrite(os.path.join(outdir, f"{tag}_mask.png"), mask)

    # 保存边缘检测结果
    if edges is not None:
        cv2.imwrite(os.path.join(outdir, f"{tag}_edges.png"), edges)

    # 绘制中心线的曲率图
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

    ax.plot(kappa, 'b-', linewidth=1, label='Centerline curvature')

    # 标记曲率峰值
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(kappa, distance=20, prominence=0.01)
    if len(peaks) > 0:
        ax.scatter(peaks, kappa[peaks], color='r', s=50, zorder=5, label='Detected peaks')

    ax.set_title("Centerline Curvature κ(s)")
    ax.set_xlabel("sample index")
    ax.set_ylabel("curvature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_curvature.png"), dpi=300)
    plt.close()

# -----------------------------
# 7) ROI 交互选择（matplotlib 3.10 兼容）
# -----------------------------
def select_rois(img_rgb):
    rois = []
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title("拖动鼠标画矩形选择 ROI；Delete 撤销；Enter/Esc 结束")
    drawn = []
    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, y1 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
        x2, y2 = int(max(eclick.xdata, erelease.ydata)), int(max(eclick.ydata, erelease.ydata))
        x2, y2 = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))
        rois.append((x1, y1, x2, y2))
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5, color='y')
        drawn.append(rect); ax.add_patch(rect); fig.canvas.draw_idle()
    selector = RectangleSelector(
        ax, onselect, useblit=False, button=[1],
        minspanx=5, minspany=5, spancoords='pixels',
        interactive=True, drag_from_anywhere=True
    )
    def on_key(event):
        if event.key in ('backspace', 'delete'):
            if rois:
                rois.pop()
            if drawn:
                rect = drawn.pop(); rect.remove(); fig.canvas.draw_idle()
        elif event.key in ('enter', 'escape'):
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return rois

# -----------------------------
# 8) 主流程
# -----------------------------
def process_image(image_path: str,
                  outdir: str,
                  resample_L: int = 512):
    # outdir 自动递增
    parent = os.path.dirname(outdir)
    name = os.path.basename(outdir.rstrip("\\/")) or "out"
    outdir = next_available_outdir(parent if parent else ".", name)
    mkdir_p(outdir)

    (img_rgb, img_alpha) = imread_rgb(image_path)
    rois = select_rois(img_rgb)

    csv_rows = []
    roi_id = 0

    for (x1, y1, x2, y2) in rois:
        roi_id += 1
        roi = img_rgb[y1:y2, x1:x2].copy()
        alpha_roi = None if img_alpha is None else img_alpha[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        # --- Mask ---
        mask = binarize_roi(roi, img_alpha=alpha_roi, black_on_white=True)

        # --- Extract edges and centerline ---
        try:
            edge1_xy, edge2_xy, centerline_xy, tip_point, root_point, edges = \
                extract_edges_and_centerline(roi, mask, resample_L=resample_L)
        except Exception as e:
            print(f"[ROI {roi_id}] edge extraction failed: {e}")
            continue

        # --- Find 4 joints ---
        try:
            joints, kappa = find_four_joints(centerline_xy, root_point, tip_point)
        except Exception as e:
            print(f"[ROI {roi_id}] joint detection failed: {e}")
            continue

        tag = f"roi{roi_id:02d}"

        # --- Visualize ---
        visualize_claw_edges(
            roi_rgb=roi, mask=mask, edge1_xy=edge1_xy, edge2_xy=edge2_xy,
            centerline_xy=centerline_xy, joints=joints, kappa=kappa,
            outdir=outdir, tag=tag, overlay_scale=2, edges=edges
        )

        # --- Compute metrics ---
        # 计算4个关节点之间的距离和角度
        row = {"roi_id": roi_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

        # 4个关节点坐标
        for i, joint in enumerate(joints):
            row[f"joint{i+1}_x"] = round(float(joint[0]), 3)
            row[f"joint{i+1}_y"] = round(float(joint[1]), 3)

        # 计算边长（4条边：0->1, 1->2, 2->3, 3->0）
        edge_lengths = []
        for i in range(4):
            p1 = joints[i]
            p2 = joints[(i + 1) % 4]
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            edge_lengths.append(float(dist))
            row[f"edge_{i+1}_len_px"] = round(dist, 3)

        total_perimeter = sum(edge_lengths)
        row["total_perimeter_px"] = round(total_perimeter, 3)

        # 计算对角线长度
        diag1 = np.hypot(joints[2][0] - joints[0][0], joints[2][1] - joints[0][1])
        diag2 = np.hypot(joints[3][0] - joints[1][0], joints[3][1] - joints[1][1])
        row["diagonal_1_px"] = round(float(diag1), 3)
        row["diagonal_2_px"] = round(float(diag2), 3)

        # 计算4个角的角度
        for i in range(4):
            p_prev = joints[(i - 1) % 4]
            p_curr = joints[i]
            p_next = joints[(i + 1) % 4]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            # 计算角度
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            angle = float(math.degrees(math.atan2(cross, dot)))
            row[f"angle_{i+1}_deg"] = round(angle, 2)

        # 中心线最大曲率值
        row["centerline_max_curvature"] = round(float(np.max(kappa)), 6)
        row["centerline_mean_curvature"] = round(float(np.mean(kappa)), 6)

        csv_rows.append(row)

    # ---- Save CSV（若被占用自动换名）----
    if csv_rows:
        base = os.path.join(outdir, "claw_metrics")
        csv_path = base + ".csv"
        try:
            f = open(csv_path, "w", newline="", encoding="utf-8")
        except PermissionError:
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = f"{base}_{ts}.csv"
            f = open(csv_path, "w", newline="", encoding="utf-8")
        with f:
            headers = sorted({k for r in csv_rows for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved metrics CSV: {csv_path}")
    else:
        print("No ROI processed successfully.")

    print(f"Done. Output dir: {outdir}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Analyze claw-shaped wave tips by extracting edge curves and finding 4 joints")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--outdir", default="out", help="Output directory (will auto-increment if exists)")
    ap.add_argument("--resample_L", type=int, default=512, help="Number of samples for edge curve resampling (recommended: 512 or 1024)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_image(
        image_path=args.image,
        outdir=args.outdir,
        resample_L=args.resample_L
    )
