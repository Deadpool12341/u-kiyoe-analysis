"""
Batch centerline extraction for multiple claw images.

This script processes all PNG images in a specified directory by calling
the centerline extraction function from extract_centerline.py.
"""

import glob
import os
import shutil
from extract_centerline import extract_centerline


# Configuration
INPUT_DIR = r"E:\python work\claw analysis\claws1"
PAD = 30
EXPORT_DIR = os.path.join(INPUT_DIR, "export")


def main():
    """Process all PNG images in the input directory."""

    print("=" * 60)
    print("BATCH CENTERLINE EXTRACTION")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Padding: {PAD} pixels")
    print(f"Export directory: {EXPORT_DIR}\n")

    # Create export directory
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Find all PNG files
    png_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else 9999
    )

    if not png_files:
        print("ERROR: No PNG files found!")
        return

    print(f"Found {len(png_files)} PNG images\n")

    # Process each image
    results = []
    for i, image_path in enumerate(png_files, 1):
        filename = os.path.basename(image_path)
        print(f"[{i}/{len(png_files)}] Processing: {filename}")

        result = extract_centerline(image_path, pad=PAD, verbose=False)
        results.append(result)

        if result['success']:
            print(f"  -> Success! robust={result['centerline_points']} pts, direct={result['centerline_points_direct']} pts")

            # Copy centerline.png (robust) to export folder
            centerline_src = os.path.join(result['output_dir'], "centerline.png")
            centerline_dst = os.path.join(EXPORT_DIR, f"{filename.replace('.png', '')}_centerline.png")
            if os.path.exists(centerline_src):
                shutil.copy2(centerline_src, centerline_dst)

            # Copy centerline_direct.png to export folder
            centerline_direct_src = os.path.join(result['output_dir'], "centerline_direct.png")
            centerline_direct_dst = os.path.join(EXPORT_DIR, f"{filename.replace('.png', '')}_centerline_direct.png")
            if os.path.exists(centerline_direct_src):
                shutil.copy2(centerline_direct_src, centerline_direct_dst)

            print(f"  -> Exported both centerlines\n")
        else:
            print(f"  -> Failed: {result['error']}\n")

    # Print summary
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count

    print("=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")

    if failed_count > 0:
        print("\nFailed files:")
        for i, result in enumerate(results):
            if not result['success']:
                filename = os.path.basename(png_files[i])
                print(f"  - {filename}: {result['error']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
