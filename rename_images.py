import os
import re
from pathlib import Path

FOLDER = r"E:\python work\claw analysis\claws1"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main():
    folder = Path(FOLDER)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: natural_key(p.name))

    if not files:
        print("No image files found.")
        return

    temp_files = []
    for i, p in enumerate(files, start=1):
        tmp = p.with_name(f"__tmp__{i:06d}{p.suffix.lower()}")
        p.rename(tmp)
        temp_files.append(tmp)

    for i, p in enumerate(temp_files, start=1):
        new_path = folder / f"{i}{p.suffix.lower()}"
        p.rename(new_path)

    print(f"Renamed {len(temp_files)} files into 1..{len(temp_files)} (keeping extensions)")

if __name__ == "__main__":
    main()
