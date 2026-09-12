import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import joblib
import warnings
import gc
import sys
import platform
import sklearn
import skimage

warnings.filterwarnings('ignore')


# 4. Segmentation quantitative validation FINAL

# ===================== 全局配置 =====================
TARGET_SIZE = (1300, 1300)

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

RANDOM_SEED = 42
SAMPLE_RATIO = 0.1
PROB_THRESHOLD = 0.85

INDOOR_CV_FOLDS = 5

EXPECTED_INDOOR_ANNOTATIONS = 20
EXPECTED_FIELD_ANNOTATIONS = 7


# 1. Preprocessing

def resize_and_pad(image,
                   target_size=TARGET_SIZE):
    """等比例缩放并居中填充。"""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        raise ValueError(
            "输入 image 尺寸无效。"
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


def mask_resize_and_pad(mask,
                        target_size=TARGET_SIZE):
    """人工 mask 使用 nearest-neighbor resize/pad。"""
    h, w = mask.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        raise ValueError(
            "输入 mask 尺寸无效。"
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
    """LAB L-channel CLAHE。"""
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


def preprocess_image(img_bgr):
    """
    Indoor 和 Field 最终统一基础 preprocessing：

    resize/pad -> CLAHE -> Gaussian blur

    Field 不再使用 Reinhard。
    """
    if img_bgr is None:
        raise ValueError(
            "输入 image 为空。"
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
        grad_mag
    )


# 2. Pixel features

def extract_pixel_features(img_rgb,
                           lab,
                           grad_mag):
    """13-dimensional pixel features，与代码 1 / 2 完全一致。"""
    h, w = img_rgb.shape[:2]

    r, g, b = cv2.split(
        img_rgb
    )

    l, a_lab, b_lab = cv2.split(
        lab
    )

    hsv = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2HSV
    )

    h_hsv, s_hsv, v_hsv = cv2.split(
        hsv
    )

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

    return np.dstack([
        r, g, b,
        l, a_lab, b_lab,
        h_hsv, s_hsv, v_hsv,
        grad_mag,
        lbp,
        x_coords,
        y_coords
    ])


# 3. Data organization

def find_matching_pairs(image_dir,
                        mask_dir,
                        suffix="_mask",
                        mask_ext=".png"):
    """匹配 image 与人工 mask。"""
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

        mask_file = (
            mask_dir /
            f"{img_file.stem}{suffix}{mask_ext}"
        )

        if mask_file.exists():

            img_paths.append(
                str(img_file)
            )

            mask_paths.append(
                str(mask_file)
            )

    return (
        img_paths,
        mask_paths
    )


def build_indoor_training_data(img_paths,
                               mask_paths,
                               sample_ratio=SAMPLE_RATIO,
                               random_state=RANDOM_SEED):
    """
    Indoor image-level CV：
    只从当前 fold 的 training images 抽取 pixels。
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

        img_bgr = cv2.imread(
            img_path
        )

        if img_bgr is None:
            raise ValueError(
                f"无法读取 image：{img_path}"
            )

        img_rgb, lab, grad_mag = (
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
                f"无法读取 mask：{mask_path}"
            )

        mask = mask_resize_and_pad(
            mask
        )

        y_all = (
            mask > 127
        ).astype(np.uint8).flatten()

        X_all = features.reshape(
            -1,
            features.shape[-1]
        )

        n_sample = max(
            1,
            int(
                len(y_all)
                * sample_ratio
            )
        )

        idx = rng.choice(
            len(y_all),
            n_sample,
            replace=False
        )

        X_list.append(
            X_all[idx]
        )

        y_list.append(
            y_all[idx]
        )

    return (
        np.vstack(X_list),
        np.hstack(y_list)
    )


# 4. RF model

def train_indoor_rf(X,
                    y):
    """与代码 1 完全一致的 RF 参数。"""
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

    return clf


# 5. FINAL mask prediction

def predict_mask(img_bgr,
                 classifier):
    """
    FINAL mask:
    probability > 0.85

    无任何额外后处理。
    """
    img_rgb, lab, grad_mag = (
        preprocess_image(
            img_bgr
        )
    )

    features = extract_pixel_features(
        img_rgb,
        lab,
        grad_mag
    )

    X = features.reshape(
        -1,
        features.shape[-1]
    )

    proba = classifier.predict_proba(
        X
    )[:, 1]

    mask = (
        proba > PROB_THRESHOLD
    ).astype(np.uint8)

    return (
        mask.reshape(
            features.shape[:2]
        )
        * 255
    )


# 6. Metrics

def load_gt_mask(mask_path):
    """Ground-truth mask：resize/pad + binary threshold。"""
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(
            f"无法读取 GT mask：{mask_path}"
        )

    mask = mask_resize_and_pad(
        mask
    )

    return (
        mask > 127
    ).astype(np.uint8)


def segmentation_metrics(gt_mask,
                         pred_mask):
    """以 needle foreground 为 positive class 计算指标。"""
    gt = (
        gt_mask > 0
    ).astype(np.uint8).flatten()

    pred = (
        pred_mask > 0
    ).astype(np.uint8).flatten()

    tp = int(
        np.sum(
            (gt == 1)
            & (pred == 1)
        )
    )

    tn = int(
        np.sum(
            (gt == 0)
            & (pred == 0)
        )
    )

    fp = int(
        np.sum(
            (gt == 0)
            & (pred == 1)
        )
    )

    fn = int(
        np.sum(
            (gt == 1)
            & (pred == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1_dice = (
        2 * tp /
        (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    iou = (
        tp /
        (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1_Dice": f1_dice,
        "IoU": iou,
        "Accuracy": accuracy,
        "GT_foreground_fraction": float(
            np.mean(
                gt == 1
            )
        ),
        "Pred_foreground_fraction": float(
            np.mean(
                pred == 1
            )
        )
    }


def summarize_metrics(df,
                      method_name):
    """输出 image-level mean ± SD 及 pooled metrics。"""
    metric_cols = [
        "Precision",
        "Recall",
        "F1_Dice",
        "IoU",
        "Accuracy"
    ]

    row = {
        "Method": method_name,
        "N_images": len(df)
    }

    for metric in metric_cols:

        values = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

        row[
            f"{metric}_mean"
        ] = values.mean()

        row[
            f"{metric}_SD"
        ] = values.std(
            ddof=1
        )

        row[
            f"{metric}_median"
        ] = values.median()

        row[
            f"{metric}_min"
        ] = values.min()

        row[
            f"{metric}_max"
        ] = values.max()

    tp = df["TP"].sum()
    tn = df["TN"].sum()
    fp = df["FP"].sum()
    fn = df["FN"].sum()

    row[
        "Pooled_Precision"
    ] = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    row[
        "Pooled_Recall"
    ] = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    row[
        "Pooled_F1_Dice"
    ] = (
        2 * tp /
        (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    row[
        "Pooled_IoU"
    ] = (
        tp /
        (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    row[
        "Pooled_Accuracy"
    ] = (
        (tp + tn) /
        (tp + tn + fp + fn)
    )

    return pd.DataFrame(
        [row]
    )


def save_validation_mask(output_dir,
                         image_name,
                         pred_mask,
                         gt_mask):
    """保存 prediction 与 GT，便于人工核查。"""
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = Path(
        image_name
    ).stem

    cv2.imwrite(
        str(
            output_dir /
            f"{stem}_pred.png"
        ),
        pred_mask
    )

    cv2.imwrite(
        str(
            output_dir /
            f"{stem}_gt.png"
        ),
        gt_mask * 255
    )


# 7. Indoor image-level 5-fold CV

def run_indoor_cv(indoor_origin_dir,
                  indoor_label_dir,
                  output_dir):

    print("\n" + "=" * 80)
    print("Indoor segmentation validation")
    print("Image-level 5-fold CV")
    print("=" * 80)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    img_paths, mask_paths = (
        find_matching_pairs(
            indoor_origin_dir,
            indoor_label_dir
        )
    )

    n_images = len(
        img_paths
    )

    print(
        f"✅ Indoor annotated images = {n_images}"
    )

    if n_images != EXPECTED_INDOOR_ANNOTATIONS:

        warnings.warn(
            f"Indoor annotations = {n_images}, "
            f"expected = {EXPECTED_INDOOR_ANNOTATIONS}"
        )

    indices = np.arange(
        n_images
    )

    kf = KFold(
        n_splits=INDOOR_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    all_results = []
    split_records = []

    for fold_id, (
        train_idx,
        test_idx
    ) in enumerate(
        kf.split(indices),
        start=1
    ):

        print(
            f"\nIndoor fold {fold_id}/{INDOOR_CV_FOLDS}"
        )

        train_imgs = [
            img_paths[i]
            for i in train_idx
        ]

        train_masks = [
            mask_paths[i]
            for i in train_idx
        ]

        test_imgs = [
            img_paths[i]
            for i in test_idx
        ]

        test_masks = [
            mask_paths[i]
            for i in test_idx
        ]

        split_records.append({
            "Fold": fold_id,
            "Training_images": "; ".join(
                Path(x).name
                for x in train_imgs
            ),
            "Test_images": "; ".join(
                Path(x).name
                for x in test_imgs
            )
        })

        X_train, y_train = (
            build_indoor_training_data(
                train_imgs,
                train_masks,
                sample_ratio=SAMPLE_RATIO,
                random_state=RANDOM_SEED
            )
        )

        clf = train_indoor_rf(
            X_train,
            y_train
        )

        fold_dir = (
            output_dir /
            f"Fold_{fold_id}"
        )

        for img_path, mask_path in zip(
            test_imgs,
            test_masks
        ):

            image_name = Path(
                img_path
            ).name

            img_bgr = cv2.imread(
                img_path
            )

            pred_mask = predict_mask(
                img_bgr,
                clf
            )

            gt_mask = load_gt_mask(
                mask_path
            )

            metrics = segmentation_metrics(
                gt_mask,
                pred_mask
            )

            metrics.update({
                "Scene": "Indoor",
                "Method": "Indoor 5-fold CV",
                "Fold": fold_id,
                "Image": image_name
            })

            all_results.append(
                metrics
            )

            save_validation_mask(
                fold_dir,
                image_name,
                pred_mask,
                gt_mask
            )

        del X_train, y_train, clf
        gc.collect()

    df = pd.DataFrame(
        all_results
    )

    first_cols = [
        "Scene",
        "Method",
        "Fold",
        "Image"
    ]

    metric_cols = [
        "Precision",
        "Recall",
        "F1_Dice",
        "IoU",
        "Accuracy",
        "GT_foreground_fraction",
        "Pred_foreground_fraction",
        "TP",
        "TN",
        "FP",
        "FN"
    ]

    df = df[
        first_cols +
        metric_cols
    ]

    summary = summarize_metrics(
        df,
        "Indoor 5-fold CV"
    )

    df.to_csv(
        output_dir /
        "Indoor_segmentation_FINAL_per_image.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        output_dir /
        "Indoor_segmentation_FINAL_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        split_records
    ).to_csv(
        output_dir /
        "Indoor_FINAL_CV_folds.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nIndoor FINAL summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    return (
        df,
        summary
    )


# 8. Independent Field validation

def run_field_independent_validation(field_origin_dir,
                                     field_label_dir,
                                     indoor_model_path,
                                     output_dir):
    """
    7 张 Field manual masks 仅用于 independent validation。

    不进行 Field retraining。
    """
    print("\n" + "=" * 80)
    print("Field segmentation independent validation")
    print("Direct Indoor-trained RF -> Field")
    print("=" * 80)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    indoor_clf = joblib.load(
        indoor_model_path
    )

    img_paths, mask_paths = (
        find_matching_pairs(
            field_origin_dir,
            field_label_dir
        )
    )

    n_images = len(
        img_paths
    )

    print(
        f"✅ Field annotated images = {n_images}"
    )

    if n_images != EXPECTED_FIELD_ANNOTATIONS:

        warnings.warn(
            f"Field annotations = {n_images}, "
            f"expected = {EXPECTED_FIELD_ANNOTATIONS}"
        )

    all_results = []

    for img_path, mask_path in zip(
        img_paths,
        mask_paths
    ):

        image_name = Path(
            img_path
        ).name

        print(
            f"   validating: {image_name}"
        )

        img_bgr = cv2.imread(
            img_path
        )

        if img_bgr is None:
            raise ValueError(
                f"无法读取 Field image：{img_path}"
            )

        pred_mask = predict_mask(
            img_bgr,
            indoor_clf
        )

        gt_mask = load_gt_mask(
            mask_path
        )

        metrics = segmentation_metrics(
            gt_mask,
            pred_mask
        )

        metrics.update({
            "Scene": "Field",
            "Method": "Independent Field validation",
            "Image": image_name
        })

        all_results.append(
            metrics
        )

        save_validation_mask(
            output_dir /
            "Predictions",
            image_name,
            pred_mask,
            gt_mask
        )

    df = pd.DataFrame(
        all_results
    )

    first_cols = [
        "Scene",
        "Method",
        "Image"
    ]

    metric_cols = [
        "Precision",
        "Recall",
        "F1_Dice",
        "IoU",
        "Accuracy",
        "GT_foreground_fraction",
        "Pred_foreground_fraction",
        "TP",
        "TN",
        "FP",
        "FN"
    ]

    df = df[
        first_cols +
        metric_cols
    ]

    summary = summarize_metrics(
        df,
        "Independent Field validation"
    )

    df.to_csv(
        output_dir /
        "Field_segmentation_FINAL_per_image.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        output_dir /
        "Field_segmentation_FINAL_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nField FINAL summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    return (
        df,
        summary
    )


# 9. Environment

def save_environment_info(output_root):
    """保存版本和 FINAL validation settings。"""
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"OpenCV: {cv2.__version__}",
        f"pandas: {pd.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"scikit-image: {skimage.__version__}",
        f"joblib: {joblib.__version__}",
        "",
        "FINAL segmentation validation settings:",
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
        f"INDOOR_CV_FOLDS = {INDOOR_CV_FOLDS}",
        "Field segmentation model = FINAL Indoor-trained RF",
        "Field annotations = independent validation only",
        "Field-specific RF retraining = none",
        "Reinhard color transfer = none",
        "Post-processing = none",
        "Final mask = raw probability-threshold mask"
    ]

    with open(
        output_root /
        "Segmentation_validation_FINAL_environment.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# Main

if __name__ == "__main__":

    # ===================== 路径配置 =====================

    INDOOR_ORIGIN_DIR = (
        "0-Indoor_origin"
    )

    INDOOR_LABEL_DIR = (
        "1-Indoor_labels"
    )

    FIELD_ORIGIN_DIR = (
        "0-Field_origin"
    )

    FIELD_LABEL_DIR = (
        "1-Field_labels"
    )

    # 代码 1 输出的最终 Indoor RF
    INDOOR_MODEL_PATH = (
        "segmentation_rf_FINAL.pkl"
    )

    OUTPUT_ROOT = Path(
        "Segmentation_quantitative_validation_FINAL"
    )

    INDOOR_OUTPUT_DIR = (
        OUTPUT_ROOT /
        "Indoor_5fold_CV"
    )

    FIELD_OUTPUT_DIR = (
        OUTPUT_ROOT /
        "Field_independent_validation"
    )

    print("=" * 80)
    print("Segmentation quantitative validation FINAL PIPELINE")
    print("=" * 80)

    # A. Indoor image-level 5-fold CV
    indoor_per_image, indoor_summary = (
        run_indoor_cv(
            indoor_origin_dir=INDOOR_ORIGIN_DIR,
            indoor_label_dir=INDOOR_LABEL_DIR,
            output_dir=INDOOR_OUTPUT_DIR
        )
    )

    # B. Independent Field validation
    field_per_image, field_summary = (
        run_field_independent_validation(
            field_origin_dir=FIELD_ORIGIN_DIR,
            field_label_dir=FIELD_LABEL_DIR,
            indoor_model_path=INDOOR_MODEL_PATH,
            output_dir=FIELD_OUTPUT_DIR
        )
    )

    # C. Combined summary
    combined_summary = pd.concat(
        [
            indoor_summary,
            field_summary
        ],
        ignore_index=True
    )

    combined_summary.to_csv(
        OUTPUT_ROOT /
        "Segmentation_validation_FINAL_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    save_environment_info(
        OUTPUT_ROOT
    )

    print("\n" + "=" * 80)
    print("FINAL segmentation validation completed")
    print("=" * 80)

    print(
        "\n正文建议优先报告："
    )

    print(
        "Precision, Recall, F1/Dice, IoU 的 image-level mean ± SD。"
    )

    print(
        "7 张 Field manual masks 仅作为 independent validation。"
    )
