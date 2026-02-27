import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg GUI backend (suitable for Windows)
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
from pathlib import Path

# ==================== SETTINGS ====================
IMAGE_DIR  = r"D:\New folder (3)\data\images"
LABELS_DIR = r"D:\New folder (3)\data\labels"
MAX_DISPLAY_EDGE = 800
CLASSES = [("Femoral Head", 0), ("Acetabular Cup", 1)]  # Classes to label in order

# Preset for thin and smooth edge overlay
EDGE_KW = dict(
    alpha=0.45,
    thicken=0,
    use_sobel=True,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    bilateral_d=7,
    bilateral_sigma=(50, 50),
    unsharp_amount=1.0,
    tophat_ksize=0,
    bothat_ksize=0,
    canny_sigma=0.33,
    edge_color=(0, 200, 255)
)
# =================================================

def normalize_params_from_display_to_original(cx_d, cy_d, a_d, b_d, w_d, h_d, w0, h0):
    """
    Convert ellipse parameters from DISPLAY coordinates back to original resolution
    and normalize in YOLO-circle format: (x/w0, y/h0, r/((w0+h0)/2))
    where r ≈ (a0 + b0) / 2
    """
    sx = w0 / float(w_d)
    sy = h0 / float(h_d)
    cx0 = cx_d * sx
    cy0 = cy_d * sy
    a0  = a_d  * sx
    b0  = b_d  * sy

    r0 = (a0 + b0) / 2.0
    r_scale = (w0 + h0) / 2.0

    x = cx0 / w0
    y = cy0 / h0
    r = r0 / r_scale
    return x, y, r

def build_edge_overlay(
    img_bgr,
    alpha=0.45,
    thicken=0,
    use_sobel=True,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    bilateral_d=7,
    bilateral_sigma=(50, 50),
    unsharp_amount=1.0,
    tophat_ksize=0,
    bothat_ksize=0,
    canny_sigma=0.33,
    edge_color=(0, 200, 255)  # BGR
):
    """
    VISUAL ONLY: Does not modify the original image,
    only overlays thin and smooth colored edges.
    """
    # Convert to LAB and enhance contrast using CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, A, B])
    enh = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Optional bilateral filtering to smooth noise
    if bilateral_d and bilateral_d > 0:
        sigmaColor, sigmaSpace = bilateral_sigma
        enh = cv2.bilateralFilter(enh, d=bilateral_d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)

    # Optional morphology (top-hat / black-hat)
    if tophat_ksize and tophat_ksize > 1:
        k = tophat_ksize | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    if bothat_ksize and bothat_ksize > 1:
        k = bothat_ksize | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gray = cv2.add(gray, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))

    # Unsharp masking to enhance edges
    if unsharp_amount and unsharp_amount > 0:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = cv2.addWeighted(gray, 1 + unsharp_amount, blur, -unsharp_amount, 0)

    # Canny edge detection
    v = np.median(gray)
    lower = int(max(0, (1.0 - canny_sigma) * v))
    upper = int(min(255, (1.0 + canny_sigma) * v))
    edges = cv2.Canny(gray, lower, upper)

    # Optional combination with Sobel magnitude
    if use_sobel:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mag_bin = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)
        edges = cv2.bitwise_or(edges, mag_bin)

    # Optional edge thickening
    if thicken and thicken > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=thicken)

    # Overlay colored edges onto the original image
    color_edges = np.zeros_like(img_bgr)
    color_edges[edges > 0] = edge_color  # BGR
    blended = cv2.addWeighted(img_bgr, 1.0, color_edges, alpha, 0.0)
    return blended

def draw_ellipse_roi(img_for_display, title):
    """
    Define an ellipse in a Matplotlib window.
    ENTER/SPACE = confirm, ESC = cancel.
    Returns: (cx, cy, a, b) in display coordinates.
    """
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img_for_display, cv2.COLOR_BGR2RGB))
    ax.set_title(title)

    params = {"cx": None, "cy": None, "a": None, "b": None}
    canceled = {"flag": False}

    # Called when the user draws the ellipse box
    def on_select(eclick, erelease):
        x1, y1 = float(eclick.xdata), float(eclick.ydata)
        x2, y2 = float(erelease.xdata), float(erelease.ydata)
        params["cx"] = (x1 + x2) / 2.0
        params["cy"] = (y1 + y2) / 2.0
        params["a"]  = abs(x2 - x1) / 2.0
        params["b"]  = abs(y2 - y1) / 2.0

    # Keyboard controls
    def on_key(event):
        if event.key in ("enter", " "):
            plt.close(fig)
        elif event.key == "escape":
            canceled["flag"] = True
            plt.close(fig)

    _ = EllipseSelector(
        ax, on_select, useblit=True,
        props=dict(edgecolor="lime", facecolor="none", linewidth=2),
        interactive=True
    )
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"\n→ {title}: Drag to draw, adjust handles, ENTER/SPACE to confirm (ESC: skip).")
    plt.show()

    if canceled["flag"]:
        return None
    if params["cx"] is None:
        raise RuntimeError(f"{title} ellipse was not selected.")
    return params["cx"], params["cy"], params["a"], params["b"]

def main():
    # Ensure label directory exists
    os.makedirs(LABELS_DIR, exist_ok=True)

    # Collect image paths
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths += glob.glob(os.path.join(IMAGE_DIR, ext))
    image_paths.sort()

    # Check if images exist
    if not image_paths:
        print(f"[X] No images found: {IMAGE_DIR}")
        return

    # Process each image
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Could not open: {img_path}")
            continue

        h0, w0 = img.shape[:2]

        # Scale only for display (annotation still uses original size)
        scale = min(MAX_DISPLAY_EDGE / max(h0, w0), 1.0)
        w_d, h_d = int(w0 * scale), int(h0 * scale)
        img_disp = cv2.resize(img, (w_d, h_d), interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()

        # Build visual edge overlay
        img_overlay = build_edge_overlay(img_disp, **EDGE_KW)

        print(f"\n🔍 {os.path.basename(img_path)} — display: {w_d}×{h_d} (original: {w0}×{h0})")
        try:
            annos = []
            for title, cls_id in CLASSES:
                sel = draw_ellipse_roi(img_overlay, title)
                if sel is None:
                    print("↩️  Image skipped.")
                    annos = None
                    break

                cx_d, cy_d, a_d, b_d = sel

                # Convert back to normalized YOLO circle
                x, y, r = normalize_params_from_display_to_original(
                    cx_d, cy_d, a_d, b_d, w_d, h_d, w0, h0
                )

                # Clamp small numeric overflow
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                r = min(max(r, 0.0), 1.0)

                annos.append((cls_id, x, y, r))

        except RuntimeError as e:
            print(f"❌ {e} — skipping.")
            continue

        if not annos:
            continue

        # Save labels
        stem = Path(img_path).stem
        out_txt = Path(LABELS_DIR) / f"{stem}.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            for cls, x, y, r in annos:
                f.write(f"{cls} {x:.6f} {y:.6f} {r:.6f}\n")

        print(f"✅ Saved: {out_txt}")

    print("\n✓ Done.")
    plt.close("all")

if __name__ == "__main__":
    main()
