"""
Utility functions for centerline extraction from claw images.
Contains skeleton extraction, pruning, and path finding algorithms.
"""

import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize, remove_small_objects


# ====== Helper Functions for Image Processing ======

def get_8_neighbors_values(img: np.ndarray, y: int, x: int):
    """
    Get 8-neighborhood pixel values in clockwise order starting from top.

    Args:
        img: Binary image array
        y: Row coordinate
        x: Column coordinate

    Returns:
        List of 8 pixel values [p2, p3, p4, p5, p6, p7, p8, p9]
    """
    h, w = img.shape

    # P2: top neighbor
    p2 = img[y-1, x]   if y > 0     else 0
    # P3: top-right neighbor
    p3 = img[y-1, x+1] if y > 0 and x < w-1 else 0
    # P4: right neighbor
    p4 = img[y,   x+1] if x < w-1  else 0
    # P5: bottom-right neighbor
    p5 = img[y+1, x+1] if y < h-1 and x < w-1 else 0
    # P6: bottom neighbor
    p6 = img[y+1, x]   if y < h-1  else 0
    # P7: bottom-left neighbor
    p7 = img[y+1, x-1] if y < h-1 and x > 0 else 0
    # P8: left neighbor
    p8 = img[y,   x-1] if x > 0    else 0
    # P9: top-left neighbor
    p9 = img[y-1, x-1] if y > 0 and x > 0 else 0

    return [p2, p3, p4, p5, p6, p7, p8, p9]


def get_8_neighbors_coords(h: int, w: int, y: int, x: int):
    """
    Get coordinates of all 8 neighbors of pixel (y, x) within image bounds.

    Args:
        h: Image height
        w: Image width
        y: Row coordinate
        x: Column coordinate

    Returns:
        List of (ny, nx) tuples for valid neighbors
    """
    neighbors = []

    # Iterate through 3x3 neighborhood
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            # Skip center pixel
            if dy == 0 and dx == 0:
                continue

            # Calculate neighbor coordinates
            ny, nx = y + dy, x + dx

            # Check if neighbor is within image bounds
            if 0 <= ny < h and 0 <= nx < w:
                neighbors.append((ny, nx))

    return neighbors


def count_skeleton_neighbors(skel: np.ndarray, y: int, x: int):
    """
    Count how many neighbors of pixel (y, x) are skeleton pixels.

    Args:
        skel: Binary skeleton image
        y: Row coordinate
        x: Column coordinate

    Returns:
        Number of non-zero neighbors (0-8)
    """
    h, w = skel.shape
    count = 0

    # Get all valid neighbor coordinates
    neighbors = get_8_neighbors_coords(h, w, y, x)

    # Count non-zero neighbors
    for ny, nx in neighbors:
        if skel[ny, nx] > 0:
            count += 1

    return count


def bfs_farthest_node(adj: list, start_idx: int):
    """
    BFS to find the farthest node from start in a graph.

    Args:
        adj: Adjacency list representation of graph
        start_idx: Starting node index

    Returns:
        Tuple of (farthest_node_index, predecessor_map)
    """
    # Initialize BFS queue
    q = deque([start_idx])

    # Distance from start to each node
    dist = {start_idx: 0}

    # Predecessor map for path reconstruction
    prev = {start_idx: None}

    # Run BFS
    while q:
        # Dequeue current node
        u = q.popleft()

        # Visit all neighbors
        for v in adj[u]:
            # If not visited yet
            if v not in dist:
                # Update distance
                dist[v] = dist[u] + 1
                # Record predecessor
                prev[v] = u
                # Add to queue
                q.append(v)

    # Find the farthest node (max distance)
    far = max(dist, key=lambda k: dist[k])

    return far, prev


# ====== Main Algorithm Functions ======

def zhang_suen_thinning(bin_img: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning algorithm to extract skeleton from binary image.

    Args:
        bin_img: Binary image with values 0 or 255

    Returns:
        Thinned skeleton image with values 0 or 255
    """
    # Convert to binary (0 or 1) for processing
    img = (bin_img > 0).astype(np.uint8)
    changed = True  # Flag to track if any pixels were removed
    h, w = img.shape  # Get image dimensions

    # Iterate until no more pixels can be removed
    while changed:
        changed = False
        to_white = []  # List of pixels to remove in this iteration

        # Step 1: Check all pixels for removal conditions
        for y in range(1, h-1):
            for x in range(1, w-1):
                # Skip if pixel is already background
                if img[y, x] == 0:
                    continue

                # Get 8 neighbors using helper function
                ns = get_8_neighbors_values(img, y, x)

                # C: Number of 0-1 transitions in neighborhood (connectivity number)
                C = sum((ns[i] == 0 and ns[(i+1) % 8] == 1) for i in range(8))

                # N: Number of non-zero neighbors
                N = sum(ns)

                # Check Zhang-Suen step 1 conditions:
                # 1. 2 <= N <= 6 (not isolated, not fully surrounded)
                # 2. C == 1 (exactly one connected component in neighborhood)
                # 3. P2*P4*P6 == 0 (at least one of top, right, bottom is background)
                # 4. P4*P6*P8 == 0 (at least one of right, bottom, left is background)
                if 2 <= N <= 6 and C == 1 and ns[0]*ns[2]*ns[4] == 0 and ns[2]*ns[4]*ns[6] == 0:
                    to_white.append((y, x))

        # Remove marked pixels
        if to_white:
            changed = True
            for y, x in to_white:
                img[y, x] = 0

        to_white = []  # Reset for step 2

        # Step 2: Check all pixels with different conditions
        for y in range(1, h-1):
            for x in range(1, w-1):
                # Skip if pixel is already background
                if img[y, x] == 0:
                    continue

                # Get 8 neighbors using helper function
                ns = get_8_neighbors_values(img, y, x)

                # C: Number of 0-1 transitions
                C = sum((ns[i] == 0 and ns[(i+1) % 8] == 1) for i in range(8))

                # N: Number of non-zero neighbors
                N = sum(ns)

                # Check Zhang-Suen step 2 conditions (slightly different from step 1):
                # 1. 2 <= N <= 6
                # 2. C == 1
                # 3. P2*P4*P8 == 0 (at least one of top, right, left is background)
                # 4. P2*P6*P8 == 0 (at least one of top, bottom, left is background)
                if 2 <= N <= 6 and C == 1 and ns[0]*ns[2]*ns[6] == 0 and ns[0]*ns[4]*ns[6] == 0:
                    to_white.append((y, x))

        # Remove marked pixels
        if to_white:
            changed = True
            for y, x in to_white:
                img[y, x] = 0

    # Convert back to 0/255 format
    return (img * 255).astype(np.uint8)


def robust_skeleton(binary_img, min_branch_length=40, border_size=8, verbose=False):
    """
    2025 永不翻车版 robust_skeleton —— 无警告、无黑图、无误杀
    """
    h, w = binary_img.shape
    orig_white = np.sum(binary_img > 0)
    if verbose:
        print(f"  Input: {h}x{w}, white pixels: {orig_white}")

    # Step 1: 智能防贴边（只在真的贴边时才动手）
    border = border_size
    if (np.any(binary_img[:border, :]>0) or np.any(binary_img[-border:,:]>0) or
        np.any(binary_img[:,:border]>0) or np.any(binary_img[:,-border:]>0)):
        mask = np.ones_like(binary_img, bool)
        mask[:border, :] = mask[-border:, :] = mask[:, :border] = mask[:, -border:] = False
        clean = binary_img.copy()
        clean[~mask] = 0
        if verbose:
            print(f"  Border cleaned: removed {orig_white - np.sum(clean>0)} pixels")
    else:
        clean = binary_img.copy()

    # Step 2: 骨架化
    skeleton = skeletonize(clean > 0)                     # 输出就是 bool

    # Step 3: 超级安全去毛刺（永别警告！）
    if min_branch_length > 0:
        before = np.sum(skeleton)
        skeleton = remove_small_objects(skeleton, min_size=min_branch_length, connectivity=1)  # 现在 100% 是 bool
        after = np.sum(skeleton)
        if before > 0 and after / before < 0.3:           # 删太多 = 误杀爪子
            if verbose:
                print(f"  Safety revert: {before}→{after}, too much removed!")
            skeleton = skeletonize(clean > 0)
        if verbose:
            print(f"  After pruning: {np.sum(skeleton)} pixels")

    # Step 4: 最后保险（骨架彻底没了就强行救活）
    if np.sum(skeleton) == 0 and orig_white > 200:
        if verbose:
            print("  Emergency recovery: using largest connected component")
        n, labels = cv2.connectedComponents(binary_img)
        if n > 1:
            largest = np.argmax(np.bincount(labels.flat)[1:]) + 1
            skeleton = skeletonize(labels == largest)

    return skeleton.astype(np.uint8) * 255

def extract_centerline_distance_transform(filled_region: np.ndarray):
    """
    Extract centerline using distance transform method.
    More accurate than direct thinning for irregular shapes.

    Args:
        filled_region: Binary image of filled region (255=foreground, 0=background)

    Returns:
        Tuple of (skeleton, distance_transform)
    """
    # Compute distance transform: each pixel value = distance to nearest background pixel
    dist_transform = cv2.distanceTransform(filled_region, cv2.DIST_L2, 5)

    # Normalize distance values to 0-255 range for visualization
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
    dist_uint8 = dist_transform.astype(np.uint8)

    # Use adaptive thresholding to extract ridge (local maxima) from distance transform
    # This captures points that are furthest from boundaries = centerline
    # Block size 15: neighborhood for local threshold computation
    # C=-2: constant subtracted from mean (negative = more aggressive threshold)
    threshold = cv2.adaptiveThreshold(dist_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, -2)

    # Further refine the skeleton using Zhang-Suen thinning
    # This ensures we get a 1-pixel wide centerline
    skel = zhang_suen_thinning(threshold)

    return skel, dist_transform


def prune_skeleton(skel: np.ndarray, min_branch_length: int = 5) -> np.ndarray:
    """
    Intelligently prune short spurious branches from skeleton while preserving real endpoints.

    Args:
        skel: Binary skeleton image
        min_branch_length: Branches shorter than this (in pixels) will be removed

    Returns:
        Pruned skeleton image
    """
    # Create a copy to avoid modifying original
    skel = skel.copy()
    h, w = skel.shape

    # Find all branch points (degree >= 3) and endpoints (degree == 1)
    branch_points = set()  # Pixels where skeleton branches (3+ neighbors)
    endpoints = set()      # Terminal pixels of skeleton (1 neighbor)

    # Scan all pixels in skeleton
    for y in range(1, h-1):
        for x in range(1, w-1):
            # Check if this is a skeleton pixel
            if skel[y, x] > 0:
                # Count how many neighbors are also skeleton pixels
                neighbor_count = count_skeleton_neighbors(skel, y, x)

                # If 3+ neighbors, this is a branch point
                if neighbor_count >= 3:
                    branch_points.add((y, x))
                # If exactly 1 neighbor, this is an endpoint
                elif neighbor_count == 1:
                    endpoints.add((y, x))

    # For each endpoint, measure distance to nearest branch point
    # Short distances indicate spurious branches that should be removed
    to_remove = set()  # Set of pixels to delete

    # Process each endpoint
    for ey, ex in endpoints:
        # Use BFS to find nearest branch point
        visited = {(ey, ex)}  # Track visited pixels
        queue = deque([(ey, ex, 0)])  # Queue: (y, x, distance)
        found_branch = False

        while queue:
            y, x, dist = queue.popleft()

            # If we've traveled too far, this is a real endpoint, stop searching
            if dist > min_branch_length:
                break

            # If we reached a branch point, this is a spurious short branch
            if (y, x) in branch_points:
                found_branch = True

                # Mark all pixels on this short branch for deletion
                # Use BFS to trace back from endpoint to branch point
                curr = (ey, ex)
                path_len = 0
                temp_visited = {curr}
                temp_queue = deque([curr])

                # Trace the branch
                while temp_queue and path_len <= dist:
                    cy, cx = temp_queue.popleft()
                    # Mark for deletion
                    to_remove.add((cy, cx))
                    path_len += 1

                    # Add neighbors to continue tracing
                    neighbors = get_8_neighbors_coords(h, w, cy, cx)
                    for ny, nx in neighbors:
                        if skel[ny, nx] > 0 and (ny, nx) not in temp_visited:
                            # Don't include the branch point itself
                            if (ny, nx) != (y, x):
                                temp_visited.add((ny, nx))
                                temp_queue.append((ny, nx))
                break

            # Continue BFS search
            neighbors = get_8_neighbors_coords(h, w, y, x)
            for ny, nx in neighbors:
                if skel[ny, nx] > 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx, dist + 1))

    # Execute deletion of spurious branches
    for y, x in to_remove:
        skel[y, x] = 0

    return skel


def extract_longest_path_from_skeleton(skel: np.ndarray):
    """
    Extract the longest connected path from a skeleton image.
    Uses graph diameter algorithm (two BFS passes).

    Args:
        skel: Binary skeleton image

    Returns:
        List of (x, y) coordinates representing the longest path
    """
    # Find all skeleton pixels
    ys, xs = np.where(skel > 0)

    # Return empty path if no skeleton pixels found
    if len(xs) == 0:
        return []

    # Build graph representation
    nodes = {(int(x), int(y)) for x, y in zip(xs, ys)}  # Set of all skeleton pixels
    pts = list(nodes)  # Convert to list for indexing: index -> (x,y)
    index_of = {p: i for i, p in enumerate(pts)}  # Reverse map: (x,y) -> index

    # Define 8-neighborhood offsets
    nbrs = [(-1, -1), (0, -1), (1, -1),  # Top row: top-left, top, top-right
            (1, 0), (1, 1), (0, 1),       # Middle & bottom row: right, bottom-right, bottom
            (-1, 1), (-1, 0)]             # Bottom-left, left

    # Build adjacency list representation of skeleton graph
    adj = [[] for _ in range(len(pts))]  # adj[i] = list of neighbor indices

    for i, (x, y) in enumerate(pts):
        # Check all 8 neighbors
        for dx, dy in nbrs:
            q = (x + dx, y + dy)  # Neighbor coordinates
            j = index_of.get(q, None)  # Get neighbor index
            # If neighbor exists in skeleton, add edge
            if j is not None:
                adj[i].append(j)

    # Find endpoints (degree 1 nodes) to use as starting points
    degrees = [len(adj[i]) for i in range(len(pts))]
    endpoints = [i for i, d in enumerate(degrees) if d == 1]

    # If no endpoints (e.g., circular skeleton), start from arbitrary node
    if not endpoints:
        endpoints = [0]

    # Tree diameter algorithm: run BFS twice
    # First BFS: from arbitrary endpoint, find one end of diameter
    a0 = endpoints[0]
    a, _ = bfs_farthest_node(adj, a0)

    # Second BFS: from that end, find the other end
    # This gives us the longest path (diameter)
    b, prev = bfs_farthest_node(adj, a)

    # Reconstruct path from b to a using predecessor map
    path_idx = []
    cur = b
    while cur is not None:
        path_idx.append(cur)
        cur = prev.get(cur, None)

    # Reverse to get path from a to b
    path_idx.reverse()

    # Convert indices back to (x, y) coordinates
    path = [pts[i] for i in path_idx]

    return path
