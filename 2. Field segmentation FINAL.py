import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern
import joblib
import warnings
import sys
import platform
import sklearn
import skimage

warnings.filterwarnings('ignore')


# 2. Field segmentation FINAL - DIRECT INDOOR RF
# FINAL segmentation pipeline:
# Indoor-trained RF
#       ↓
# Field image
#       ↓
# resize/pad
#       ↓
# CLAHE
#       ↓
# Gaussian blur
#       ↓
# 13-dimensional pixel features
#       ↓
# Indoor RF prediction
#       ↓
# probability > 0.85
#       ↓
# FINAL Field mask

# ===================== 全局配置 =====================
TARGET_SIZE = (1300, 1300)   # (宽, 高)

CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

GAUSSIAN_KERNEL = (3, 3)
GAUSSIAN_SIGMA = 0.5

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = 'uniform'

IMAGE_EXTS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tiff'
}

PROB_THRESHOLD = 0.85


# 1. Field image preprocessing

def resize_and_pad(image,
                   target_size=TARGET_SIZE):
    """等比例缩放并居中填充至 TARGET_SIZE。"""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        raise ValueError(
            "输入图像尺寸无效。"
        )

    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8
    )

    x_offset = (
        target_w - new_w
    ) // 2

    y_offset = (
        target_h - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


def clahe_lab(image_bgr,
              clip_limit=CLAHE_CLIP,
              tile_grid_size=CLAHE_GRID):
    """在 LAB color space 中对 L channel 执行 CLAHE。"""
    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_enhanced = clahe.apply(
        l
    )

    lab_enhanced = cv2.merge([
        l_enhanced,
        a,
        b
    ])

    return cv2.cvtColor(
        lab_enhanced,
        cv2.COLOR_LAB2BGR
    )


def gaussian_blur(image_bgr,
                  kernel=GAUSSIAN_KERNEL,
                  sigma=GAUSSIAN_SIGMA):
    """Gaussian filtering。"""
    return cv2.GaussianBlur(
        image_bgr,
        kernel,
        sigma
    )


def preprocess_field_image(img_bgr):
    """
    FINAL Field preprocessing：

    resize/pad -> CLAHE -> Gaussian blur

    不使用 Reinhard color transfer。
    """
    if img_bgr is None:
        raise ValueError(
            "输入 Field image 为空。"
        )

    img_bgr = resize_and_pad(
        img_bgr
    )

    img_bgr = clahe_lab(
        img_bgr
    )

    img_bgr = gaussian_blur(
        img_bgr
    )

    lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB
    )

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    grad_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1, 0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0, 1,
        ksize=3
    )

    grad_mag = cv2.magnitude(
        grad_x,
        grad_y
    )

    if grad_mag.max() > 0:

        grad_mag = np.uint8(
            np.clip(
                grad_mag /
                grad_mag.max() *
                255,
                0,
                255
            )
        )

    else:

        grad_mag = np.zeros_like(
            gray,
            dtype=np.uint8
        )

    return (
        img_rgb,
        lab,
        grad_mag,
        img_bgr
    )


# 2. 13-dimensional pixel features

def extract_pixel_features(img_rgb,
                           lab,
                           grad_mag):
    """
    与 Indoor segmentation FINAL 完全一致的 13-dimensional pixel features：

    RGB(3)
    + LAB(3)
    + HSV(3)
    + Sobel gradient(1)
    + LBP(1)
    + normalized x/y coordinates(2)
    """
    h, w = img_rgb.shape[:2]

    # RGB
    r, g, b = cv2.split(
        img_rgb
    )

    # LAB
    l, a_lab, b_lab = cv2.split(
        lab
    )

    # HSV
    hsv = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2HSV
    )

    h_hsv, s_hsv, v_hsv = cv2.split(
        hsv
    )

    # LBP
    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    lbp = local_binary_pattern(
        gray,
        LBP_POINTS,
        LBP_RADIUS,
        method=LBP_METHOD
    )

    lbp = np.uint8(
        np.clip(
            lbp *
            (255.0 / (LBP_POINTS + 1)),
            0,
            255
        )
    )

    # normalized spatial coordinates
    x_coords = (
        np.tile(
            np.arange(w),
            (h, 1)
        ).astype(np.float32)
        / w
    )

    y_coords = (
        np.tile(
            np.arange(h).reshape(-1, 1),
            (1, w)
        ).astype(np.float32)
        / h
    )

    features = np.dstack([
        r, g, b,
        l, a_lab, b_lab,
        h_hsv, s_hsv, v_hsv,
        grad_mag,
        lbp,
        x_coords,
        y_coords
    ])

    return features


# 3. Load Indoor-trained RF

def load_indoor_model(model_path):
    """加载代码 1 输出的 FINAL Indoor RF segmentation model。"""
    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到 Indoor RF model：{model_path}"
        )

    clf = joblib.load(
        model_path
    )

    print(
        f"✅ Indoor-trained RF loaded: {model_path}"
    )

    return clf


# 4. Direct Field segmentation

def segment_field_image(img_bgr,
                        classifier,
                        threshold=PROB_THRESHOLD):
    """
    FINAL Field segmentation：

    probability > 0.85 -> foreground(255)

    不进行任何额外后处理。
    """
    img_rgb, lab, grad_mag, img_bgr_processed = (
        preprocess_field_image(
            img_bgr
        )
    )

    features = extract_pixel_features(
        img_rgb,
        lab,
        grad_mag
    )

    feat_flat = features.reshape(
        -1,
        features.shape[-1]
    )

    proba = classifier.predict_proba(
        feat_flat
    )[:, 1]

    pred_mask = (
        proba > threshold
    ).astype(np.uint8)

    pred_mask = pred_mask.reshape(
        features.shape[:2]
    ) * 255

    return (
        pred_mask,
        img_rgb,
        img_bgr_processed
    )


# 5. Batch segmentation

def batch_process_field(input_dir,
                        preprocess_dir,
                        segmentation_dir,
                        classifier,
                        prob_threshold=PROB_THRESHOLD):
    """
    批量处理全部 Field images。

    保存：
    1) FINAL Field preprocessed images
    2) FINAL direct segmentation masks
    3) overlays
    """
    input_dir = Path(
        input_dir
    )

    preprocess_dir = Path(
        preprocess_dir
    )

    segmentation_dir = Path(
        segmentation_dir
    )

    preprocess_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    segmentation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        raise FileNotFoundError(
            f"在 {input_dir} 中未找到 Field images。"
        )

    foreground_records = []

    for img_file in image_files:

        print(
            f"处理：{img_file.name}"
        )

        img_bgr = cv2.imread(
            str(img_file)
        )

        if img_bgr is None:

            print(
                f"⚠️ 跳过 {img_file.name}：读取失败"
            )

            continue

        pred_mask, img_rgb, img_bgr_processed = (
            segment_field_image(
                img_bgr,
                classifier,
                threshold=prob_threshold
            )
        )

        # 保存 FINAL preprocessed image
        cv2.imwrite(
            str(
                preprocess_dir /
                img_file.name
            ),
            img_bgr_processed
        )

        # 保存 FINAL mask
        cv2.imwrite(
            str(
                segmentation_dir /
                f"{img_file.stem}_mask.png"
            ),
            pred_mask
        )

        # overlay
        overlay = img_rgb.copy()

        overlay[
            pred_mask > 0
        ] = (
            0,
            255,
            0
        )

        overlay_img = cv2.addWeighted(
            img_rgb,
            0.7,
            overlay,
            0.3,
            0
        )

        cv2.imwrite(
            str(
                segmentation_dir /
                f"{img_file.stem}_overlay.jpg"
            ),
            cv2.cvtColor(
                overlay_img,
                cv2.COLOR_RGB2BGR
            )
        )

        foreground_fraction = float(
            np.mean(
                pred_mask > 0
            )
        )

        foreground_records.append(
            (
                img_file.name,
                foreground_fraction
            )
        )

        print(
            f"   ✓ foreground fraction = {foreground_fraction:.4f}"
        )

    print(
        f"\n✅ FINAL Field segmentation completed: {len(foreground_records)} images"
    )

    return foreground_records


# 6. Environment

def save_environment_info(output_path):
    """保存软件版本与 FINAL pipeline 参数。"""
    lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"OpenCV: {cv2.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"scikit-image: {skimage.__version__}",
        f"joblib: {joblib.__version__}",
        "",
        "FINAL Field segmentation settings:",
        f"TARGET_SIZE = {TARGET_SIZE}",
        f"CLAHE_CLIP = {CLAHE_CLIP}",
        f"CLAHE_GRID = {CLAHE_GRID}",
        f"GAUSSIAN_KERNEL = {GAUSSIAN_KERNEL}",
        f"GAUSSIAN_SIGMA = {GAUSSIAN_SIGMA}",
        f"LBP_POINTS = {LBP_POINTS}",
        f"LBP_RADIUS = {LBP_RADIUS}",
        f"LBP_METHOD = {LBP_METHOD}",
        f"PROB_THRESHOLD = {PROB_THRESHOLD}",
        "Segmentation model = Indoor-trained RF",
        "Field-specific RF retraining = none",
        "Reinhard color transfer = none",
        "Post-processing = none",
        "Final mask = raw probability-threshold mask"
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# Main

if __name__ == "__main__":

    # ===================== 路径配置 =====================

    FIELD_ORIGIN_DIR = (
        "0-Field_origin"
    )

    # 来自代码 1：
    INDOOR_MODEL_PATH = (
        "segmentation_rf_FINAL.pkl"
    )

    # FINAL Field preprocessed images
    FIELD_PREPROCESS_DIR = (
        "1-Field_preprocess_FINAL"
    )

    # FINAL Field masks
    FIELD_SEGMENTATION_DIR = (
        "2-Field_Segmentation_FINAL"
    )

    ENVIRONMENT_FILE = (
        "Field_segmentation_FINAL_environment.txt"
    )

    print("=" * 80)
    print("Field segmentation FINAL - DIRECT INDOOR RF")
    print("NO REINHARD / NO FIELD RETRAINING / NO POST-PROCESSING")
    print("=" * 80)

    # 1. Load FINAL Indoor-trained RF
    indoor_clf = load_indoor_model(
        INDOOR_MODEL_PATH
    )

    # 2. Directly segment all Field images
    batch_process_field(
        input_dir=FIELD_ORIGIN_DIR,
        preprocess_dir=FIELD_PREPROCESS_DIR,
        segmentation_dir=FIELD_SEGMENTATION_DIR,
        classifier=indoor_clf,
        prob_threshold=PROB_THRESHOLD
    )

    # 3. Save environment
    save_environment_info(
        ENVIRONMENT_FILE
    )

    print("\n✨ FINAL direct Field segmentation completed")
    print(
        f"   - Field preprocess: {FIELD_PREPROCESS_DIR}"
    )
    print(
        f"   - Field masks: {FIELD_SEGMENTATION_DIR}"
    )
