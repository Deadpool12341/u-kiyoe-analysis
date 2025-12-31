"""
Refactored Centerline Extraction Module

Improved readability and modularity for easier maintenance.
"""

import os
import cv2
import numpy as np
from utils import zhang_suen_thinning, robust_skeleton, extract_longest_path_from_skeleton


# ============ STEP 1: Image Loading ============

def load_and_prepare_image(input_path, pad=25):
    """Load grayscale image and add padding."""
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return gray, padded


# ============ STEP 2: Binarization ============

def binarize_and_invert(gray_padded):
    """Apply Otsu binarization and invert to get white boundaries on black background."""
    _, binary = cv2.threshold(gray_padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    edges_thick = cv2.dilate(inverted, np.ones((2, 2), np.uint8), iterations=1)
    return binary, inverted, edges_thick


# ============ STEP 3: Fill Mask Generation ============

def fill_region_between_curves(closed, pad, out_dir, verbose=True):
    """Fill the region between claw boundaries using progressive morphological closing."""
    h, w = closed.shape

    # Define kernels with odd sizes
    def odd_size(val):
        return val + 1 if val % 2 == 0 else val

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    medium_sz = odd_size(min(2 * pad - 20, 25))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (medium_sz, medium_sz))

    large_sz = odd_size(min(2 * pad - 10, 45))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (large_sz, large_sz))

    extra_large_sz = odd_size(min(2 * pad - 5, 55))
    kernel_extra_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_large_sz, extra_large_sz))

    if verbose:
        print("Performing progressive morphological closing...")

    # Progressive closing
    closed_region = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_small)
    closed_region = cv2.morphologyEx(closed_region, cv2.MORPH_CLOSE, kernel_medium)
    closed_region = cv2.morphologyEx(closed_region, cv2.MORPH_CLOSE, kernel_large)
    closed_region = cv2.morphologyEx(closed_region, cv2.MORPH_CLOSE, kernel_extra_large, iterations=2)

    # Fill contours
    inside = np.zeros((h, w), np.uint8)
    contours, _ = cv2.findContours(closed_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(inside, contours, -1, 255, thickness=cv2.FILLED)
    cv2.imwrite(os.path.join(out_dir, "filled_mask.png"), inside)

    return inside


# ============ STEP 4: Skeleton Extraction ============

def extract_skeletons(inside, out_dir, verbose=True):
    """Extract both robust and direct skeletons from filled mask."""
    if verbose:
        print("Extracting skeletons...")

    skel_robust = robust_skeleton(inside, min_branch_length=40, border_size=8)
    skel_direct = zhang_suen_thinning(inside)

    cv2.imwrite(os.path.join(out_dir, "skeleton_robust.png"), skel_robust)
    cv2.imwrite(os.path.join(out_dir, "skeleton_direct.png"), skel_direct)
    cv2.imwrite(os.path.join(out_dir, "skeleton.png"), skel_robust)

    return skel_robust, skel_direct


# ============ STEP 5: Visualization ============

def visualize_and_save_results(gray, pad, path_robust, path_direct, skel_robust, skel_direct, out_dir):
    """Create visualizations for both robust and direct centerlines, and save coordinates."""

    # Visualization 1: Robust skeleton + centerline
    vis_robust = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(skel_robust > 0)
    for x, y in zip(xs, ys):
        vis_robust[y, x] = (0, 255, 0)
    for i in range(1, len(path_robust)):
        cv2.line(vis_robust, path_robust[i - 1], path_robust[i], (0, 0, 255), 1, cv2.LINE_AA)
    vis_robust = vis_robust[pad:-pad, pad:-pad]
    cv2.imwrite(os.path.join(out_dir, "centerline.png"), vis_robust)

    # Visualization 2: Direct skeleton + centerline
    vis_direct = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(skel_direct > 0)
    for x, y in zip(xs, ys):
        vis_direct[y, x] = (0, 255, 0)
    for i in range(1, len(path_direct)):
        cv2.line(vis_direct, path_direct[i - 1], path_direct[i], (0, 0, 255), 1, cv2.LINE_AA)
    vis_direct = vis_direct[pad:-pad, pad:-pad]
    cv2.imwrite(os.path.join(out_dir, "centerline_direct.png"), vis_direct)

    # Save robust centerline coordinates
    if len(path_robust) > 0:
        path_array = np.array(path_robust) - pad
        np.savetxt(
            os.path.join(out_dir, "centerline.csv"),
            path_array,
            delimiter=',',
            header='x,y',
            comments='',
            fmt='%d'
        )

    # Save direct centerline coordinates
    if len(path_direct) > 0:
        path_array = np.array(path_direct) - pad
        np.savetxt(
            os.path.join(out_dir, "centerline_direct.csv"),
            path_array,
            delimiter=',',
            header='x,y',
            comments='',
            fmt='%d'
        )


# ============ STEP 6: Main Function ============

def extract_centerline(input_path, pad=25, verbose=True):
    """
    Extract centerline from a claw image.

    Args:
        input_path (str): Path to the input PNG image
        pad (int): Padding size in pixels (default: 25)
        verbose (bool): Whether to print progress messages (default: True)

    Returns:
        dict: Result dictionary containing:
            - 'success' (bool): Whether extraction succeeded
            - 'centerline_points' (int): Number of centerline points extracted
            - 'path' (list): List of (x,y) tuples representing the centerline
            - 'output_dir' (str): Path to output directory
            - 'error' (str): Error message if failed (None if successful)
    """
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(os.path.dirname(input_path), f"{input_filename}_output")
    os.makedirs(out_dir, exist_ok=True)

    try:
        # Step 1: Load and prepare image
        gray, padded = load_and_prepare_image(input_path, pad)

        # Step 2: Binarization
        binary, inverted, closed = binarize_and_invert(padded)

        # Save debug images
        cv2.imwrite(os.path.join(out_dir, "debug_binary.png"), binary)
        cv2.imwrite(os.path.join(out_dir, "debug_inverted.png"), inverted)
        cv2.imwrite(os.path.join(out_dir, "debug_closed.png"), closed)

        # Step 3: Fill region between curves
        inside = fill_region_between_curves(closed, pad, out_dir, verbose)

        # Step 4: Extract skeletons
        skel_robust, skel_direct = extract_skeletons(inside, out_dir, verbose)

        # Step 5: Extract longest paths from both skeletons
        path_robust = extract_longest_path_from_skeleton(skel_robust)
        path_direct = extract_longest_path_from_skeleton(skel_direct)

        # Step 6: Visualize and save results for both
        visualize_and_save_results(padded, pad, path_robust, path_direct, skel_robust, skel_direct, out_dir)

        if verbose:
            print(f"Success! Centerline points: robust={len(path_robust)}, direct={len(path_direct)}")

        return {
            'success': True,
            'centerline_points': len(path_robust),
            'centerline_points_direct': len(path_direct),
            'path': path_robust,
            'path_direct': path_direct,
            'output_dir': out_dir,
            'error': None
        }

    except Exception as e:
        if verbose:
            print(f"Error: {str(e)}")
        return {
            'success': False,
            'centerline_points': 0,
            'centerline_points_direct': 0,
            'path': [],
            'path_direct': [],
            'output_dir': out_dir,
            'error': str(e)
        }


# ============ STEP 7: Standalone Run ============

if __name__ == "__main__":
    DEFAULT_INPUT = r"E:\python work\claw analysis\claws1\1.png"
    result = extract_centerline(DEFAULT_INPUT, pad=25, verbose=True)

    if result['success']:
        print(f"Results saved to: {result['output_dir']}")
    else:
        print(f"Failed: {result['error']}")
