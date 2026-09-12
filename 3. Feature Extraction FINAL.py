import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops
)
import warnings

warnings.filterwarnings('ignore')


# 3. Feature Extraction FINAL - DIRECT FIELD PIPELINE
# Geometry: 3
#   area
#   perimeter
#   compactness
#
# Color: 18
#   RGB mean / variance / skewness
#   LAB mean / variance / skewness
#
# Texture: 13
#   LBP bins 0--9
#   GLCM contrast / energy / correlation
#
# Total = 3 + 18 + 13 = 34

IMAGE_EXTS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tiff'
}


# 1. Geometry features: 3

def compute_geometry(mask):
    """
    从 binary segmentation mask 中计算 3 个 geometry features。
    """
    mask_bool = (
        mask > 0
    )

    area = int(
        np.sum(
            mask_bool
        )
    )

    if area == 0:
        return {
            'area': 0,
            'perimeter': 0.0,
            'compactness': 0.0
        }

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    perimeter = float(
        sum(
            cv2.arcLength(
                cnt,
                True
            )
            for cnt in contours
        )
    )

    compactness = (
        float(
            4 *
            np.pi *
            area /
            (perimeter ** 2)
        )
        if perimeter > 0
        else 0.0
    )

    return {
        'area': area,
        'perimeter': perimeter,
        'compactness': compactness
    }


# 2. Color features: 18

def _safe_skew(channel):
    """避免 constant array 导致 skew = nan。"""
    if channel.size == 0:
        return 0.0

    if np.allclose(
        channel,
        channel[0]
    ):
        return 0.0

    val = skew(
        channel
    )

    if np.isnan(
        val
    ):
        return 0.0

    return float(
        val
    )


def compute_color_features(img_rgb,
                           mask):
    """
    在 segmentation foreground 内计算：
    RGB / LAB 各 3 channels 的
    mean / variance / skewness
    共 18 features。
    """
    img_lab = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2LAB
    )

    mask_bool = (
        mask > 0
    )

    feature_names = []

    for space, channels in [
        (
            'RGB',
            ['R', 'G', 'B']
        ),
        (
            'LAB',
            ['L', 'A', 'B']
        )
    ]:

        for name in channels:

            feature_names.extend([
                f'{space}_{name}_mean',
                f'{space}_{name}_var',
                f'{space}_{name}_skew'
            ])

    if np.sum(
        mask_bool
    ) == 0:

        return {
            key: 0.0
            for key in feature_names
        }

    pixels_rgb = img_rgb[
        mask_bool
    ]

    pixels_lab = img_lab[
        mask_bool
    ]

    features = {}

    # RGB
    for i, name in enumerate([
        'R',
        'G',
        'B'
    ]):

        channel = (
            pixels_rgb[:, i]
            .astype(np.float32)
        )

        features[
            f'RGB_{name}_mean'
        ] = float(
            np.mean(channel)
        )

        features[
            f'RGB_{name}_var'
        ] = float(
            np.var(channel)
        )

        features[
            f'RGB_{name}_skew'
        ] = _safe_skew(
            channel
        )

    # LAB
    for i, name in enumerate([
        'L',
        'A',
        'B'
    ]):

        channel = (
            pixels_lab[:, i]
            .astype(np.float32)
        )

        features[
            f'LAB_{name}_mean'
        ] = float(
            np.mean(channel)
        )

        features[
            f'LAB_{name}_var'
        ] = float(
            np.var(channel)
        )

        features[
            f'LAB_{name}_skew'
        ] = _safe_skew(
            channel
        )

    return features


# 3. Texture features: 13

def compute_texture_features(gray,
                             mask):
    """
    Texture features:
    - LBP: 10-bin histogram
    - GLCM: contrast / energy / correlation

    保持与原 feature extraction 方法一致。
    """
    base_features = {
        **{
            f'LBP_bin{i}': 0.0
            for i in range(10)
        },
        'GLCM_contrast': 0.0,
        'GLCM_energy': 0.0,
        'GLCM_correlation': 0.0
    }

    if np.sum(
        mask > 0
    ) == 0:
        return base_features

    # 背景置 0
    masked_gray = gray.copy()

    masked_gray[
        mask == 0
    ] = 0

    # ---------------------
    # LBP
    # ---------------------
    lbp = local_binary_pattern(
        masked_gray,
        P=8,
        R=1,
        method='uniform'
    )

    lbp_vals = (
        lbp[
            mask > 0
        ].astype(np.uint8)
    )

    hist, _ = np.histogram(
        lbp_vals,
        bins=np.arange(
            0,
            11
        ),
        density=True
    )

    for i in range(10):

        base_features[
            f'LBP_bin{i}'
        ] = (
            float(hist[i])
            if i < len(hist)
            else 0.0
        )

    # ---------------------
    # GLCM
    # ---------------------
    # 量化为 16 grayscale levels
    img_quant = np.uint8(
        masked_gray / 16
    )

    glcm = graycomatrix(
        img_quant,
        distances=[1],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=16,
        symmetric=True,
        normed=True
    )

    base_features[
        'GLCM_contrast'
    ] = float(
        np.mean(
            graycoprops(
                glcm,
                'contrast'
            )
        )
    )

    base_features[
        'GLCM_energy'
    ] = float(
        np.mean(
            graycoprops(
                glcm,
                'energy'
            )
        )
    )

    corr = np.mean(
        graycoprops(
            glcm,
            'correlation'
        )
    )

    base_features[
        'GLCM_correlation'
    ] = float(
        0.0
        if np.isnan(corr)
        else corr
    )

    return base_features


# 4. Single-image feature extraction

def extract_features_for_image(img_path,
                               mask_path):
    """
    对单张 FINAL preprocessed image + FINAL segmentation mask
    提取 34 features。
    """
    img_bgr = cv2.imread(
        str(img_path)
    )

    if img_bgr is None:
        raise ValueError(
            f"无法读取 image：{img_path}"
        )

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(
            f"无法读取 mask：{mask_path}"
        )

    mask = (
        mask > 127
    ).astype(np.uint8)

    if (
        mask.shape[:2]
        != img_rgb.shape[:2]
    ):

        mask = cv2.resize(
            mask,
            (
                img_rgb.shape[1],
                img_rgb.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # 3 geometry
    geometry = compute_geometry(
        mask
    )

    # 18 color
    color = compute_color_features(
        img_rgb,
        mask
    )

    # 13 texture
    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    texture = compute_texture_features(
        gray,
        mask
    )

    features = {
        **geometry,
        **color,
        **texture
    }

    if len(
        features
    ) != 34:

        raise ValueError(
            f"特征数量错误：当前 {len(features)}，应为 34。"
        )

    return features


# 5. Batch feature extraction

def extract_scene_features(scene_name,
                           img_dir,
                           mask_dir,
                           output_csv):
    """
    批量提取一个 scene 的 34 features。
    """
    img_dir = Path(
        img_dir
    )

    mask_dir = Path(
        mask_dir
    )

    image_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        raise FileNotFoundError(
            f"{scene_name}: 在 {img_dir} 中未找到 images。"
        )

    records = []

    print("\n" + "=" * 70)
    print(
        f"{scene_name} feature extraction FINAL"
    )
    print("=" * 70)

    for img_path in image_files:

        mask_path = (
            mask_dir /
            f"{img_path.stem}_mask.png"
        )

        if not mask_path.exists():

            print(
                f"⚠️ 跳过 {img_path.name}：无对应 FINAL mask"
            )

            continue

        try:

            features = (
                extract_features_for_image(
                    img_path,
                    mask_path
                )
            )

            features[
                'filename'
            ] = img_path.name

            records.append(
                features
            )

            print(
                f"✓ {img_path.name}"
            )

        except Exception as e:

            print(
                f"❌ {img_path.name}: {e}"
            )

    if not records:
        raise ValueError(
            f"{scene_name}: 未成功提取任何 features。"
        )

    df = pd.DataFrame(
        records
    )

    # 明确固定最终列顺序
    feature_order = [
        # geometry 3
        'area',
        'perimeter',
        'compactness',

        # RGB 9
        'RGB_R_mean',
        'RGB_R_var',
        'RGB_R_skew',
        'RGB_G_mean',
        'RGB_G_var',
        'RGB_G_skew',
        'RGB_B_mean',
        'RGB_B_var',
        'RGB_B_skew',

        # LAB 9
        'LAB_L_mean',
        'LAB_L_var',
        'LAB_L_skew',
        'LAB_A_mean',
        'LAB_A_var',
        'LAB_A_skew',
        'LAB_B_mean',
        'LAB_B_var',
        'LAB_B_skew',

        # LBP 10
        *[
            f'LBP_bin{i}'
            for i in range(10)
        ],

        # GLCM 3
        'GLCM_contrast',
        'GLCM_energy',
        'GLCM_correlation'
    ]

    if len(
        feature_order
    ) != 34:

        raise RuntimeError(
            "内部 feature_order 不是 34。"
        )

    missing_cols = [
        c
        for c in feature_order
        if c not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"缺失 features：{missing_cols}"
        )

    df = df[
        ['filename']
        + feature_order
    ]

    df.to_csv(
        output_csv,
        index=False,
        encoding='utf-8-sig'
    )

    print(
        f"\n✅ {scene_name}: {len(df)} images"
    )

    print(
        f"✅ 34 predictors saved to: {output_csv}"
    )

    return df


# Main

if __name__ == "__main__":

    # ----------------------------
    # Indoor
    # ----------------------------
    INDOOR_IMG_DIR = (
        "1-Indoor_preprocess_FINAL"
    )

    INDOOR_MASK_DIR = (
        "2-Indoor_Segmentation_FINAL"
    )

    INDOOR_OUTPUT_CSV = (
        "Indoor_features_FINAL.csv"
    )

    # ----------------------------
    # Field
    # ----------------------------
    FIELD_IMG_DIR = (
        "1-Field_preprocess_FINAL"
    )

    FIELD_MASK_DIR = (
        "2-Field_Segmentation_FINAL"
    )

    FIELD_OUTPUT_CSV = (
        "Field_features_FINAL.csv"
    )

    # ----------------------------
    # Run
    # ----------------------------
    indoor_df = extract_scene_features(
        scene_name="Indoor",
        img_dir=INDOOR_IMG_DIR,
        mask_dir=INDOOR_MASK_DIR,
        output_csv=INDOOR_OUTPUT_CSV
    )

    field_df = extract_scene_features(
        scene_name="Field",
        img_dir=FIELD_IMG_DIR,
        mask_dir=FIELD_MASK_DIR,
        output_csv=FIELD_OUTPUT_CSV
    )

    print("\n" + "=" * 70)
    print("FINAL feature extraction completed - DIRECT Field pipeline")
    print("=" * 70)

    print(
        f"Indoor: {len(indoor_df)} images × 34 predictors"
    )

    print(
        f"Field : {len(field_df)} images × 34 predictors"
    )
