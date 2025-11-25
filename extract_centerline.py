"""
Centerline extraction from claw images.

This script processes a claw image to extract the centerline between two curved boundaries.
Main steps:
1. Load and preprocess image
2. Detect and connect boundary curves
3. Fill region between curves
4. Extract skeleton/centerline
5. Prune spurious branches
6. Find longest path as final centerline
7. Visualize and save results
"""

import cv2
import numpy as np
import os

# Import helper functions from utils module
from utils import (
    zhang_suen_thinning,
    robust_skeleton,
    extract_centerline_distance_transform,
    prune_skeleton,
    extract_longest_path_from_skeleton
)

# ====== Configuration ======
# Path to input image (claw template)
INPUT_PATH = r"E:\python work\claw analysis\template5.png"

# Auto-generate output directory based on input image name
# Example: "template.png" -> "template_output/centerline/"
input_filename = os.path.splitext(os.path.basename(INPUT_PATH))[0]  # Get "template" from "template.png"
BASE_OUT_DIR = os.path.join(os.path.dirname(INPUT_PATH), f"{input_filename}_output")
OUT_DIR = os.path.join(BASE_OUT_DIR, "centerline")

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")


# ====== Main Processing Pipeline ======
def main():
    """
    Main processing function to extract centerline from claw image.
    """

    # ========== STEP 0: Load Image ==========
    # Read input image in grayscale mode
    gray = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)

    # Check if image was loaded successfully
    if gray is None:
        raise FileNotFoundError(f"Cannot load image: {INPUT_PATH}")

    # Add white padding around image (20 pixels on all sides)
    # This prevents boundary artifacts when processing touches image edges
    PAD = 5
    gray = cv2.copyMakeBorder(
        gray,           # Source image
        PAD,            # Top padding
        PAD,            # Bottom padding
        PAD,            # Left padding
        PAD,            # Right padding
        cv2.BORDER_CONSTANT,  # Pad with constant value
        value=255       # White padding (255 = white in grayscale)
    )


    # ========== STEP 1: Binarization ==========
    # Convert grayscale to binary using Otsu's automatic thresholding
    # Otsu's method automatically finds optimal threshold value
    _, binary = cv2.threshold(
        gray,                           # Input grayscale image
        0,                              # Threshold value (0 = auto with OTSU flag)
        255,                            # Maximum value (white)
        cv2.THRESH_BINARY + cv2.THRESH_OTSU  # Binary threshold + Otsu's method
    )

    # Invert binary image: black curves (0) become white (255), white background becomes black
    # After inversion: curves = white (255), background = black (0)
    inverted = cv2.bitwise_not(binary)


    # ========== STEP 2: Boundary Enhancement ==========
    # Dilate the inverted boundaries slightly to make them thicker
    # This helps ensure the two curves connect during morphological closing
    # Use 2x2 kernel with 1 iteration
    edges_thick = cv2.dilate(
        inverted,                   # Input: inverted binary (curves are white)
        np.ones((2, 2), np.uint8),  # Structuring element: 2x2 square
        iterations=1                # Number of times to apply dilation
    )

    # Assign to variable for clarity (no additional closing here to avoid merging the two curves)
    closed = edges_thick

    # Save debug images to visualize intermediate steps
    cv2.imwrite(os.path.join(OUT_DIR, "debug_binary.png"), binary)      # Original binary
    cv2.imwrite(os.path.join(OUT_DIR, "debug_inverted.png"), inverted)  # Inverted (curves white)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_closed.png"), closed)      # After dilation


    # ========== STEP 3: Fill Region Between Curves ==========
    # Get image dimensions
    h, w = closed.shape

    # Create three elliptical kernels of increasing size for progressive morphological closing
    # Ellipse shape is better than square for curved shapes
    # IMPORTANT: Kernel sizes are scaled based on PAD to avoid edge artifacts
    # Rule: largest kernel radius should be < PAD to operate properly at edges
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,  # Shape: ellipse
        (5, 5)              # Size: 5x5 pixels (radius ~2.5)
    )

    # Medium kernel: scale based on available padding
    medium_size = min(11, 2 * PAD - 3)  # Ensure radius < PAD
    medium_size = max(medium_size, 7)    # Minimum useful size
    if medium_size % 2 == 0:             # Must be odd
        medium_size += 1
    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,  # Shape: ellipse
        (medium_size, medium_size)  # Size: adaptive based on PAD
    )

    # Large kernel: scale based on available padding
    large_size = min(17, 2 * PAD - 1)   # Ensure radius < PAD
    large_size = max(large_size, 25)      # Minimum useful size
    if large_size % 2 == 0:              # Must be odd
        large_size += 1
    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,  # Shape: ellipse
        (large_size, large_size)  # Size: adaptive based on PAD
    )

    # Debug: print kernel sizes
    print(f"PAD={PAD}, Kernel sizes: small=5x5, medium={medium_size}x{medium_size}, large={large_size}x{large_size}")

    # Apply progressive morphological closing operations
    # Closing = dilation followed by erosion
    # This connects nearby boundaries while preserving overall shape

    # First closing: connect nearby gaps (small kernel)
    closed_region = cv2.morphologyEx(
        closed,              # Input image
        cv2.MORPH_CLOSE,     # Operation: closing
        kernel_small         # Kernel size
    )

    # Second closing: connect larger gaps (medium kernel)
    closed_region = cv2.morphologyEx(
        closed_region,       # Input: result from previous closing
        cv2.MORPH_CLOSE,     # Operation: closing
        kernel_medium        # Kernel size
    )

    # Third closing: ensure full connection between curves (large kernel)
    closed_region = cv2.morphologyEx(
        closed_region,       # Input: result from previous closing
        cv2.MORPH_CLOSE,     # Operation: closing
        kernel_large         # Kernel size
    )

    # Initialize empty mask for filled region
    inside = np.zeros((h, w), np.uint8)

    # Find external contours of the closed region
    # RETR_EXTERNAL: retrieve only outermost contours
    # CHAIN_APPROX_NONE: store all contour points
    contours_filled, _ = cv2.findContours(
        closed_region,           # Input binary image
        cv2.RETR_EXTERNAL,       # Retrieval mode: external contours only
        cv2.CHAIN_APPROX_NONE    # Approximation: no approximation, all points
    )

    # Draw filled contours on the mask
    # This creates a solid white region where the claw interior is
    cv2.drawContours(
        inside,              # Destination image (our mask)
        contours_filled,     # List of contours to draw
        -1,                  # Contour index (-1 = draw all)
        255,                 # Color: white (255)
        thickness=cv2.FILLED # Fill the contour completely
    )

    # NOTE: We do NOT subtract boundaries here
    # The skeleton algorithms (Zhang-Suen and distance transform) naturally
    # find the medial axis, so keeping the full filled region preserves
    # endpoints and ensures centerline extends to the claw opening

    # Save filled mask for debugging
    cv2.imwrite(os.path.join(OUT_DIR, "filled_mask.png"), inside)


    # ========== STEP 4: Skeleton Extraction ==========
    # Use industrial-grade robust skeleton extraction
    # This prevents edge-sticking, removes spurs, and handles screenshot borders
    print("Extracting skeleton with edge-protection...")

    skel_robust = robust_skeleton(
        inside,
        min_branch_length=40,  # Remove branches shorter than 40 pixels
        border_size=8          # Remove pixels within 8px of image edge
    )
    cv2.imwrite(os.path.join(OUT_DIR, "skeleton_robust.png"), skel_robust)

    # For comparison, also save Zhang-Suen result (legacy)
    skel_direct = zhang_suen_thinning(inside)
    cv2.imwrite(os.path.join(OUT_DIR, "skeleton_direct.png"), skel_direct)

    # Use the robust skeleton (no pruning needed - already clean)
    skel_pruned = skel_robust.copy()

    # Save final skeleton
    cv2.imwrite(os.path.join(OUT_DIR, "skeleton.png"), skel_pruned)
    print(f"Skeleton extracted with border protection (removed {8}px from edges)")


    # ========== STEP 6: Extract Longest Path ==========
    # Find the longest connected path through the skeleton
    # This represents the main centerline from claw opening to tip
    path = extract_longest_path_from_skeleton(skel_pruned)


    # ========== STEP 7: Visualization ==========
    # Create color visualization: original image with overlaid skeleton and centerline

    # Convert grayscale to BGR for colored overlay
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Find all skeleton pixels
    ys, xs = np.where(skel_pruned > 0)

    # Draw skeleton in green
    for x, y in zip(xs, ys):
        vis[y, x] = (0, 255, 0)  # BGR: green

    # Draw centerline path in red
    # Connect consecutive points with lines
    for i in range(1, len(path)):
        cv2.line(
            vis,                # Destination image
            path[i-1],          # Start point (previous point)
            path[i],            # End point (current point)
            (0, 0, 255),        # Color: BGR = red
            1,                  # Thickness: 1 pixel (note: comment says 8, but value is 1)
            cv2.LINE_AA         # Anti-aliased line for smooth appearance
        )

    # Remove padding to return to original image size
    # Crop: [top:bottom, left:right]
    vis = vis[PAD:-PAD, PAD:-PAD]

    # Save final visualization
    cv2.imwrite(os.path.join(OUT_DIR, "centerline.png"), vis)


    # ========== STEP 8: Save Centerline Coordinates ==========
    if len(path) > 0:
        # Convert path to numpy array
        path_array = np.array(path)

        # Adjust coordinates back to original image space (remove padding offset)
        path_array = path_array - PAD

        # Save to CSV file
        # Format: x,y coordinates (one point per line)
        np.savetxt(
            os.path.join(OUT_DIR, "centerline.csv"),  # Output file path
            path_array,                                # Data: Nx2 array of (x,y) points
            delimiter=',',                             # CSV delimiter
            header='x,y',                              # Column headers
            comments='',                               # No comment character
            fmt='%d'                                   # Integer format
        )


    # ========== STEP 9: Print Summary ==========
    print("Done!")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Centerline points: {len(path)}")


# ====== Entry Point ======
if __name__ == "__main__":
    main()
