#!/usr/bin/env python3
# Converts YOLO-circle (cls x y r_norm) --> YOLO-box (cls x y w_norm h_norm)
# NOTE: r_norm = r_px / ((w_px + h_px)/2)

import argparse
from pathlib import Path
from PIL import Image

# Supported image extensions
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def find_image(image_dir: Path, stem: str):
    """
    Find the corresponding image file for a given label stem.
    First tries known extensions, then falls back to any extension.
    """
    for ext in IMG_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any extension
    for p in image_dir.glob(f"{stem}.*"):
        return p
    return None

def clamp01(x: float) -> float:
    """
    Clamp a value to the [0, 1] range.
    """
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def convert_file(label_path: Path, image_dir: Path, out_dir: Path = None, backup: bool = True):
    """
    Convert a single YOLO-circle label file into YOLO-box format.
    """
    # Find matching image
    img_path = find_image(image_dir, label_path.stem)
    if img_path is None:
        print(f"⚠️  Image not found, skipped: {label_path.stem}")
        return

    # Read image size
    w_px, h_px = Image.open(img_path).size
    if w_px <= 0 or h_px <= 0:
        print(f"⚠️  Invalid image size, skipped: {img_path}")
        return

    lines_out = []
    bad = False

    # Read label file line by line
    with label_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()

            # Skip empty lines and comments
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 4:
                print(f"⚠️  {label_path.name}:{ln} -> expected 'cls x y r', line skipped: {s}")
                bad = True
                continue

            try:
                cls = int(float(parts[0]))
                x, y, r = map(float, parts[1:4])
            except Exception:
                print(f"⚠️  {label_path.name}:{ln} -> parse error, line skipped: {s}")
                bad = True
                continue

            # Convert normalized radius to pixel radius
            r_px = r * (w_px + h_px) / 2.0

            # Convert circle to YOLO bounding box (normalized)
            w_norm = (2.0 * r_px) / w_px
            h_norm = (2.0 * r_px) / h_px

            # Safety clamp for center coordinates
            x = clamp01(x)
            y = clamp01(y)

            # Width/height may exceed 1 if circle is very large
            if w_norm > 1.0 or h_norm > 1.0:
                print(f"⚠️  Large box: {label_path.name}:{ln} -> w={w_norm:.3f}, h={h_norm:.3f} (not clipped)")

            lines_out.append(f"{cls} {x:.6f} {y:.6f} {w_norm:.6f} {h_norm:.6f}")

    # If nothing valid produced
    if not lines_out and bad:
        print(f"⚠️  Empty output: {label_path.name}")
        return

    # Decide output location
    if out_dir is None:
        # Write in-place (optionally create backup first)
        if backup:
            bak = label_path.with_suffix(".txt.circle.bak")
            try:
                label_path.replace(bak)
            except Exception:
                print(f"⚠️  Backup failed: {bak}")
                # continue overwriting if backup fails
        target = label_path if not backup else (label_path)  # same path after replace()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / label_path.name

    # Write converted labels
    with target.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print(f"✅ Converted: {label_path.name}")

def main():
    """
    CLI entry point.
    """
    ap = argparse.ArgumentParser(description="YOLO-circle -> YOLO-box converter")
    ap.add_argument("--labels", default="data/labels", help="label folder (circle format)")
    ap.add_argument("--images", default="data/images", help="image folder")
    ap.add_argument("--out", default=None, help="output label folder (writes in-place if not set)")
    ap.add_argument("--no-backup", action="store_true", help="disable backup when writing in-place")
    args = ap.parse_args()

    label_dir = Path(args.labels)
    image_dir = Path(args.images)
    out_dir = None if args.out is None else Path(args.out)

    # Validate directories
    if not label_dir.exists():
        print(f"[X] Label folder not found: {label_dir}"); return
    if not image_dir.exists():
        print(f"[X] Image folder not found: {image_dir}"); return

    # Process all label files
    for p in sorted(label_dir.glob("*.txt")):
        convert_file(p, image_dir, out_dir=out_dir, backup=(not args.no_backup))

if __name__ == "__main__":
    main()
