"""
Analyze centerline curvature and place joints at key locations.

This script:
1. Loads the centerline from CSV
2. Calculates curvature along the centerline
3. Finds the two highest curvature points (natural joint locations)
4. Visualizes joints as green circles at:
   - Start point (claw opening)
   - End point (claw tip)
   - Two highest curvature points
"""

import cv2
import numpy as np
import os
from scipy.signal import savgol_filter, find_peaks


# ====== Configuration ======
INPUT_IMAGE = r"E:\python work\claw analysis\template3.png"

# Auto-generate paths based on input image name
# Example: "template.png" -> "template_output/centerline/" and "template_output/joints/"
input_filename = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]  # Get "template" from "template.png"
BASE_OUT_DIR = os.path.join(os.path.dirname(INPUT_IMAGE), f"{input_filename}_output")
CENTERLINE_CSV = os.path.join(BASE_OUT_DIR, "centerline", "centerline.csv")
OUT_DIR = os.path.join(BASE_OUT_DIR, "joints")

# Joint visualization parameters
JOINT_RADIUS = 3        # Radius of green circles (increased for visibility)
JOINT_COLOR = (0, 255, 0)  # Green color (BGR)
JOINT_THICKNESS = -1    # Filled circles (-1 = filled)

# Curvature calculation parameters (Savitzky-Golay filter)
SG_WINDOW = 21          # Window size for Savitzky-Golay filter (must be odd)
SG_POLY = 3             # Polynomial order for Savitzky-Golay filter

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Input image: {INPUT_IMAGE}")
print(f"Centerline CSV: {CENTERLINE_CSV}")
print(f"Output directory: {OUT_DIR}\n")


# ====== Helper Functions ======

def ensure_odd(n: int) -> int:
    """Ensure a number is odd (required for Savitzky-Golay filter)"""
    return n if n % 2 == 1 else n + 1


def calculate_curvature(points, sg_window: int = 21, sg_poly: int = 3):
    """
    Calculate curvature using Savitzky-Golay filter (more robust than Gaussian).
    Based on analyze_claw.py's compute_curvature function.

    Curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

    Args:
        points: Nx2 array of (x, y) coordinates
        sg_window: Savitzky-Golay filter window size (must be odd)
        sg_poly: Polynomial order for Savitzky-Golay filter

    Returns:
        Array of curvature values
    """
    # Extract x and y coordinates
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    # Ensure window size is valid
    sg_window = ensure_odd(sg_window)
    sg_window = min(sg_window, len(x) - 1)  # Can't be larger than data
    if sg_window < 5:
        sg_window = 5
    sg_window = ensure_odd(sg_window)

    # Calculate first derivatives using Savitzky-Golay filter
    dx = savgol_filter(x, sg_window, sg_poly, deriv=1)
    dy = savgol_filter(y, sg_window, sg_poly, deriv=1)

    # Calculate second derivatives
    ddx = savgol_filter(x, sg_window, sg_poly, deriv=2)
    ddy = savgol_filter(y, sg_window, sg_poly, deriv=2)

    # Calculate curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.maximum((dx**2 + dy**2)**(3/2), 1e-8)
    curvature = numerator / denominator

    return curvature


def find_curvature_peaks(curvature, num_peaks=2):
    """
    Ultimate robust peak finder designed for claws with right-angle roots.
    Aggressively blocks first 33% and last 15% to avoid boundary artifacts.

    Args:
        curvature: Array of curvature values
        num_peaks: Number of peaks to find (default=2)

    Returns:
        Indices of the top curvature peaks
    """
    n = len(curvature)

    # Aggressively block endpoints: front 33% and back 15%
    # This is specifically designed for claws with right-angle roots
    mask = np.zeros_like(curvature, dtype=bool)
    start = int(0.33 * n)
    end = int(0.85 * n)
    mask[start:end] = True

    search_region = end - start
    print(f"\nCurvature analysis (robust mode for right-angle roots):")
    print(f"  Total points: {n}")
    print(f"  Blocked: first {start} points (33%) and last {n-end} points (15%)")
    print(f"  Search region: [{start}:{end}] = {search_region} points ({search_region/n*100:.1f}%)")

    # Create masked curvature (zero out blocked regions)
    masked_curv = curvature.copy()
    masked_curv[~mask] = 0

    # Find peaks with relaxed threshold (35% of max)
    peaks, _ = find_peaks(
        masked_curv,
        height=np.max(masked_curv) * 0.35,  # Lower threshold for sensitivity
        distance=10                          # Minimum distance between peaks
    )

    print(f"\nFound {len(peaks)} curvature peaks in search region:")
    for i, peak_idx in enumerate(peaks):
        print(f"  Peak {i+1}: index={peak_idx}, curvature={curvature[peak_idx]:.4f}")

    # Select top num_peaks by curvature value
    if len(peaks) >= num_peaks:
        # Sort peaks by curvature value and take top num_peaks
        selected = peaks[np.argsort(curvature[peaks])[-num_peaks:]]
        selected = sorted(selected)
    else:
        # Fallback: take the top num_peaks points in the valid region
        print(f"  Warning: Found < {num_peaks} peaks, using fallback method")
        valid_indices = np.argsort(curvature[start:end])[-num_peaks:] + start
        selected = sorted(valid_indices)

    print(f"\nSelected top {num_peaks} peaks:")
    for i, peak_idx in enumerate(selected):
        position_pct = peak_idx / n * 100
        print(f"  Selected peak {i+1}: index={peak_idx} ({position_pct:.1f}% along path), curvature={curvature[peak_idx]:.4f}")

    return selected


# ====== Main Processing ======

def main():
    """
    Main function to analyze curvature and place joints.
    """

    # ========== STEP 1: Load Data ==========
    print("Loading data...")

    # Check if input image exists
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    # Check if centerline CSV exists
    if not os.path.exists(CENTERLINE_CSV):
        print(f"\n❌ ERROR: Centerline CSV not found!")
        print(f"   Expected: {CENTERLINE_CSV}")
        print(f"\n💡 You need to run extract_centerline.py first to generate the centerline!")
        print(f"   1. Make sure extract_centerline.py has INPUT_PATH = r'{INPUT_IMAGE}'")
        print(f"   2. Run: python extract_centerline.py")
        print(f"   3. Then run: python analyze_joints.py")
        raise FileNotFoundError(f"Centerline CSV not found: {CENTERLINE_CSV}")

    # Load original image
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {INPUT_IMAGE}")

    # IMPORTANT: Add same padding as extract_centerline.py to match coordinate system
    # The centerline CSV has coordinates in original image space (after PAD was subtracted)
    # But if those coordinates go negative, we need to add padding back
    # Read PAD value from extract_centerline.py - currently PAD=10
    PAD = 5
    img_padded = cv2.copyMakeBorder(
        img,
        PAD, PAD, PAD, PAD,
        cv2.BORDER_CONSTANT,
        value=255  # White padding
    )
    print(f"Added {PAD}px padding to image: {img.shape} -> {img_padded.shape}")

    # Load centerline coordinates from CSV
    # Format: first row is header "x,y", remaining rows are coordinates
    centerline_original = np.loadtxt(CENTERLINE_CSV, delimiter=',', skiprows=1)

    # Add PAD back to coordinates to match padded image
    centerline = centerline_original + PAD

    print(f"Loaded {len(centerline)} centerline points")
    print(f"Coordinate adjustment: added PAD={PAD} to match padded image")


    # ========== STEP 2: Calculate Curvature ==========
    print("Calculating curvature...")

    # Calculate curvature using Savitzky-Golay filter (from analyze_claw.py)
    curvature = calculate_curvature(centerline, sg_window=SG_WINDOW, sg_poly=SG_POLY)

    # Save curvature data to CSV for analysis
    curvature_data = np.column_stack([
        centerline[:len(curvature), 0],  # x coordinates
        centerline[:len(curvature), 1],  # y coordinates
        curvature                         # curvature values
    ])
    np.savetxt(
        os.path.join(OUT_DIR, "curvature.csv"),
        curvature_data,
        delimiter=',',
        header='x,y,curvature',
        comments='',
        fmt='%.6f'
    )
    print(f"Curvature range: [{curvature.min():.4f}, {curvature.max():.4f}]")


    # ========== STEP 3: Find Joint Locations ==========
    print("Finding joint locations...")
    joint_start = centerline[0]

    # Joint 1: End point (claw tip)
    joint_end = centerline[-1]

    # Joints 2 & 3: Two highest curvature points (using robust algorithm)
    # Designed for claws with right-angle roots - blocks first 33% and last 15%
    peak_indices = find_curvature_peaks(curvature, num_peaks=2)

    # Get coordinates of high-curvature joints
    joint_curve1 = centerline[peak_indices[0]]
    joint_curve2 = centerline[peak_indices[1]]

    # Store all joints (3 total: end + 2 curvature peaks)
    # Note: Start joint removed per user request
    joints = [
        ("Start(tip)", joint_start, curvature[0]),
        ("End (tip)", joint_end, curvature[-1] if len(curvature) > 0 else 0),
        ("High curvature 1", joint_curve1, curvature[peak_indices[0]]),
        ("High curvature 2", joint_curve2, curvature[peak_indices[1]])
    ]

    # Print joint information with detailed debugging
    print(f"\nCenterline length: {len(centerline)} points")
    print(f"Curvature array length: {len(curvature)} points")
    print(f"Image dimensions (with padding): {img_padded.shape}")

    print("\nJoint locations:")
    for i, (name, (x, y), curv) in enumerate(joints):
        in_bounds = 0 <= x < img_padded.shape[1] and 0 <= y < img_padded.shape[0]
        print(f"  Joint {i+1} - {name}: ({x:.1f}, {y:.1f}), curvature={curv:.4f}, in_bounds={in_bounds}")

    # Check for duplicate positions
    positions = [(int(x), int(y)) for _, (x, y), _ in joints]
    unique_positions = set(positions)
    if len(unique_positions) < len(positions):
        print(f"\n⚠ WARNING: Only {len(unique_positions)} unique positions for {len(positions)} joints!")
        print(f"  Positions: {positions}")
        for pos in unique_positions:
            count = positions.count(pos)
            if count > 1:
                print(f"  Position {pos} has {count} joints overlapping")


    # ========== STEP 4: Visualize Joints ==========
    print("\nVisualizing joints...")

    # Create visualization image using padded image
    vis = img_padded.copy()

    # Draw centerline in red (thin line so joints are visible)
    for i in range(1, len(centerline)):
        pt1 = tuple(centerline[i-1].astype(int))
        pt2 = tuple(centerline[i].astype(int))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    # Draw joints as circles with labels
    # 3 joints: end (tip) + 2 curvature peaks
    colors = [
        (255, 0, 0),
        (0, 255, 255),  # Yellow for end (tip)
        (0, 255, 0),    # Green for curvature 1
        (255, 0, 255)   # Magenta for curvature 2
    ]

    for idx, (name, (x, y), curv) in enumerate(joints):
        center = (int(x), int(y))

        # Check if coordinates are valid
        if not (0 <= center[0] < vis.shape[1] and 0 <= center[1] < vis.shape[0]):
            print(f"  ⚠ Joint {idx+1} is OUT OF BOUNDS: {center}")
            continue

        # Draw circle with unique color
        cv2.circle(vis, center, JOINT_RADIUS, colors[idx], JOINT_THICKNESS, cv2.LINE_AA)

        # Draw white outline for visibility
        cv2.circle(vis, center, JOINT_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Add label number
        label = str(idx + 1)
        cv2.putText(vis, label, (int(x) + 8, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2, cv2.LINE_AA)

        print(f"  ✓ Drawing joint {idx+1} at pixel ({center[0]}, {center[1]}) - {name}")

    # Save visualization WITH padding (don't crop, so all joints are visible)
    output_path = os.path.join(OUT_DIR, "joints_visualization.png")
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to: {output_path}")
    print(f"Output image size: {vis.shape} (with {PAD}px padding to show all joints)")

    # Also save a cropped version (may cut off edge joints)
    vis_cropped = vis[PAD:-PAD, PAD:-PAD]
    output_path_cropped = os.path.join(OUT_DIR, "joints_visualization_cropped.png")
    cv2.imwrite(output_path_cropped, vis_cropped)
    print(f"Also saved cropped version: {output_path_cropped}")


    # ========== STEP 5: Create Curvature Plot ==========
    print("Creating curvature plot...")

    # Create a visualization showing curvature along the path
    # Image height = original image height
    # Image width = number of points
    plot_height = 200
    plot_width = len(curvature)

    # Normalize curvature to plot height
    curv_normalized = curvature / curvature.max() * (plot_height - 20) if curvature.max() > 0 else curvature

    # Create plot image (white background)
    plot = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    # Draw curvature curve (blue)
    for i in range(1, len(curv_normalized)):
        pt1 = (i-1, plot_height - int(curv_normalized[i-1]) - 10)
        pt2 = (i, plot_height - int(curv_normalized[i]) - 10)
        cv2.line(plot, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)

    # Mark peak locations (green vertical lines) - only 2 peaks now
    for i, peak_idx in enumerate(peak_indices):
        if peak_idx < plot_width:
            cv2.line(plot, (peak_idx, 0), (peak_idx, plot_height), (0, 255, 0), 2)

    # Save curvature plot
    plot_path = os.path.join(OUT_DIR, "curvature_plot.png")
    cv2.imwrite(plot_path, plot)
    print(f"Saved curvature plot to: {plot_path}")


    # ========== STEP 6: Save Joint Coordinates ==========
    # Save joint coordinates to CSV (in original image coordinate space)
    joint_coords = np.array([[x, y] for _, (x, y), _ in joints])
    # Subtract PAD to get coordinates in original image space
    joint_coords_original = joint_coords - PAD
    np.savetxt(
        os.path.join(OUT_DIR, "joints.csv"),
        joint_coords_original,
        delimiter=',',
        header='x,y',
        comments='',
        fmt='%.2f'
    )
    print(f"Joint coordinates saved (in original image space, PAD={PAD} removed)")

    print("\nDone!")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
