import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
import sys
import platform
import sklearn
import skimage

warnings.filterwarnings('ignore')


# 1. Indoor segmentation FINAL
# ===================== 全局配置 =====================
TARGET_SIZE = (1300, 1300)   # (宽, 高)

CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

GAUSSIAN_KERNEL = (3, 3)
GAUSSIAN_SIGMA = 0.5

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = 'uniform'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

RANDOM_SEED = 42
SAMPLE_RATIO = 0.1
PROB_THRESHOLD = 0.85

# 1. 图像预处理
# ============================================================
def resize_and_pad(image, target_size=TARGET_SIZE):
    """等比例缩放并居中填充至目标尺寸，保持原始比例。"""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        raise ValueError("输入图像尺寸无效。")

    scale = min(target_w / w, target_h / h)

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


def mask_resize_and_pad(mask, target_size=TARGET_SIZE):
    """
    对人工 mask 执行与图像完全一致的 resize + pad。
    mask 使用 INTER_NEAREST，避免标签插值。
    """
    h, w = mask.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        raise ValueError("输入 mask 尺寸无效。")

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
        mask,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros(
        (target_h, target_w),
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
    """LAB color space 中对 L channel 进行 CLAHE。"""
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
    """Gaussian filtering."""
    return cv2.GaussianBlur(
        image_bgr,
        kernel,
        sigma
    )


def preprocess_image(img_bgr):
    """
    Indoor preprocessing:
    resize/pad -> CLAHE -> Gaussian blur

    返回：
    1) RGB image
    2) LAB image
    3) Sobel gradient magnitude
    4) 最终预处理 BGR image
    """
    if img_bgr is None:
        raise ValueError("输入图像为空。")

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
# ============================================================
def extract_pixel_features(img_rgb,
                           lab,
                           grad_mag):
    """
    13-dimensional pixel features：
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


# 3. 人工标注图像匹配
# ============================================================
def find_matching_pairs(image_dir,
                        mask_dir,
                        suffix="_mask",
                        mask_ext=".png"):
    """
    匹配原始 image 与人工 mask。

    image:
        xxx.jpg

    mask:
        xxx_mask.png
    """
    image_dir = Path(
        image_dir
    )

    mask_dir = Path(
        mask_dir
    )

    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )

    img_paths = []
    mask_paths = []

    for img_file in image_files:

        mask_name = (
            f"{img_file.stem}"
            f"{suffix}"
            f"{mask_ext}"
        )

        mask_file = (
            mask_dir /
            mask_name
        )

        if mask_file.exists():

            img_paths.append(
                str(img_file)
            )

            mask_paths.append(
                str(mask_file)
            )

    return img_paths, mask_paths


# 4. RF training data
# ============================================================
def build_training_data(img_paths,
                        mask_paths,
                        sample_ratio=SAMPLE_RATIO,
                        random_state=RANDOM_SEED):
    """
    从每张人工标注图像中随机抽取 10% pixels 作为 RF training data。
    """
    rng = np.random.default_rng(
        random_state
    )

    X_list = []
    y_list = []

    for img_path, mask_path in zip(
        img_paths,
        mask_paths
    ):

        print(
            f"处理训练样本：{Path(img_path).name}"
        )

        img_bgr = cv2.imread(
            img_path
        )

        if img_bgr is None:
            raise ValueError(
                f"无法读取图像：{img_path}"
            )

        img_rgb, lab, grad_mag, _ = (
            preprocess_image(
                img_bgr
            )
        )

        features = extract_pixel_features(
            img_rgb,
            lab,
            grad_mag
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(
                f"无法读取人工 mask：{mask_path}"
            )

        mask = mask_resize_and_pad(
            mask
        )

        mask_bin = (
            mask > 127
        ).astype(np.uint8).flatten()

        feat_flat = features.reshape(
            -1,
            features.shape[-1]
        )

        n_pixels = len(
            mask_bin
        )

        n_sample = max(
            1,
            int(
                n_pixels *
                sample_ratio
            )
        )

        indices = rng.choice(
            n_pixels,
            n_sample,
            replace=False
        )

        X_list.append(
            feat_flat[indices]
        )

        y_list.append(
            mask_bin[indices]
        )

    if not X_list:
        raise ValueError(
            "未能构建 Indoor RF training data。"
        )

    X = np.vstack(
        X_list
    )

    y = np.hstack(
        y_list
    )

    print(
        f"训练数据构建完成：X {X.shape}, y {y.shape}"
    )

    return X, y


# 5. RF segmentation model
# ============================================================
def train_segmentation_model(X,
                             y,
                             save_path):
    """训练 Indoor RF segmentation model。"""
    print(
        "\n开始训练 Indoor Random Forest segmentation model..."
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        max_samples=0.7,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight={
            0: 1,
            1: 2
        }
    )

    clf.fit(
        X,
        y
    )

    print(
        f"训练完成，training accuracy = {clf.score(X, y):.4f}"
    )

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        clf,
        save_path
    )

    print(
        f"模型已保存：{save_path}"
    )

    return clf


# 6. Final segmentation：raw probability mask
# ============================================================
def segment_image(img_bgr,
                  classifier,
                  threshold=PROB_THRESHOLD):
    """
    FINAL segmentation：

    probability > 0.85 -> foreground(255)
    otherwise -> background(0)

    不再进行 morphology / contour filling / area filtering。
    """
    img_rgb, lab, grad_mag, img_bgr_processed = (
        preprocess_image(
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


def batch_process(input_dir,
                  preprocess_dir,
                  segmentation_dir,
                  classifier,
                  prob_threshold=PROB_THRESHOLD):
    """
    批量处理全部 Indoor images。

    保存：
    1) 最终预处理图像
    2) FINAL raw binary mask
    3) overlay
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
            f"在 {input_dir} 中未找到图像。"
        )

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
            segment_image(
                img_bgr,
                classifier,
                threshold=prob_threshold
            )
        )

        # 保存最终预处理图像
        cv2.imwrite(
            str(
                preprocess_dir /
                img_file.name
            ),
            img_bgr_processed
        )

        # 保存 FINAL segmentation mask
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

        foreground_fraction = (
            np.mean(
                pred_mask > 0
            )
        )

        print(
            f"   ✓ foreground fraction = {foreground_fraction:.4f}"
        )

    print(
        f"\n批量处理完成：{segmentation_dir}"
    )

# 7. 软件环境与参数记录
# ============================================================
def save_environment_info(output_path):
    """保存 Python/package versions 和最终 segmentation 参数。"""
    lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"OpenCV: {cv2.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"scikit-image: {skimage.__version__}",
        f"joblib: {joblib.__version__}",
        "",
        "FINAL Indoor segmentation settings:",
        f"TARGET_SIZE = {TARGET_SIZE}",
        f"CLAHE_CLIP = {CLAHE_CLIP}",
        f"CLAHE_GRID = {CLAHE_GRID}",
        f"GAUSSIAN_KERNEL = {GAUSSIAN_KERNEL}",
        f"GAUSSIAN_SIGMA = {GAUSSIAN_SIGMA}",
        f"LBP_POINTS = {LBP_POINTS}",
        f"LBP_RADIUS = {LBP_RADIUS}",
        f"LBP_METHOD = {LBP_METHOD}",
        f"SAMPLE_RATIO = {SAMPLE_RATIO}",
        f"PROB_THRESHOLD = {PROB_THRESHOLD}",
        f"RANDOM_SEED = {RANDOM_SEED}",
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
# ============================================================
if __name__ == "__main__":

    # ===================== 路径配置 =====================
    ORIGIN_DIR = "0-Indoor_origin"

    # 20 张人工标注图像对应的 masks
    LABEL_DIR = "1-Indoor_labels"

    # 保存与 segmentation 完全一致的最终预处理图像
    PREPROCESS_DIR = "1-Indoor_preprocess_FINAL"

    # 保存最终 raw binary masks + overlays
    SEGMENTATION_DIR = "2-Indoor_Segmentation_FINAL"

    # FINAL Indoor RF model
    MODEL_PATH = "segmentation_rf_FINAL.pkl"

    # 参数记录
    ENVIRONMENT_FILE = "Indoor_segmentation_FINAL_environment.txt"

    print("=" * 75)
    print("Indoor needle segmentation FINAL")
    print("Raw probability-threshold mask; no contour filling")
    print("=" * 75)

    # 1. 找到人工标注数据
    img_paths, mask_paths = (
        find_matching_pairs(
            image_dir=ORIGIN_DIR,
            mask_dir=LABEL_DIR,
            suffix="_mask",
            mask_ext=".png"
        )
    )

    if len(img_paths) == 0:
        raise FileNotFoundError(
            "未找到任何 Indoor annotated images。"
        )

    print(
        f"✅ 找到 {len(img_paths)} 张 Indoor annotated images"
    )

    # 2. 构建 RF training pixels
    X_train, y_train = (
        build_training_data(
            img_paths,
            mask_paths,
            sample_ratio=SAMPLE_RATIO,
            random_state=RANDOM_SEED
        )
    )

    # 3. 训练最终 Indoor RF
    clf = train_segmentation_model(
        X_train,
        y_train,
        save_path=MODEL_PATH
    )

    # 4. 批量分割全部 Indoor images
    batch_process(
        input_dir=ORIGIN_DIR,
        preprocess_dir=PREPROCESS_DIR,
        segmentation_dir=SEGMENTATION_DIR,
        classifier=clf,
        prob_threshold=PROB_THRESHOLD
    )

    # 5. 保存软件环境
    save_environment_info(
        ENVIRONMENT_FILE
    )

    print("\n✨ Indoor FINAL segmentation completed")
    print(
        f"   - Model: {MODEL_PATH}"
    )
    print(
        f"   - Preprocessed images: {PREPROCESS_DIR}"
    )
    print(
        f"   - Final masks: {SEGMENTATION_DIR}"
    )
