import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# ==================== 配置参数 ====================
IMG_DIR = "1-Field_preprocess"          # 预处理后的野外图像
MASK_DIR = "2-Field_Segmentation"       # 野外分割掩膜
OUTPUT_CSV = "Field_features.csv"       # 输出特征文件

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# ==================== 特征提取函数 ====================
def compute_geometry(mask):
    """从二值掩膜计算几何特征和密度特征（共4个）"""
    area = int(np.sum(mask > 0))
    if area == 0:
        return {
            'area': 0,
            'perimeter': 0.0,
            'compactness': 0.0,
            'density': 0.0
        }

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    compactness = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    density = float(area / mask.size)

    return {
        'area': area,
        'perimeter': perimeter,
        'compactness': compactness,
        'density': density
    }


def _safe_skew(channel: np.ndarray) -> float:
    """避免常数数组导致 skew 返回 nan"""
    if channel.size == 0:
        return 0.0
    if np.allclose(channel, channel[0]):
        return 0.0
    val = skew(channel)
    return float(0.0 if np.isnan(val) else val)


def compute_color_features(img_rgb, mask):
    """计算针叶区域的 RGB 和 LAB 颜色统计量（共18个）"""
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    mask_bool = mask > 0

    feature_names = []
    for space, channels in [('RGB', ['R', 'G', 'B']), ('LAB', ['L', 'A', 'B'])]:
        for name in channels:
            feature_names.extend([
                f'{space}_{name}_mean',
                f'{space}_{name}_var',
                f'{space}_{name}_skew'
            ])

    if np.sum(mask_bool) == 0:
        return {k: 0.0 for k in feature_names}

    pixels_rgb = img_rgb[mask_bool]
    pixels_lab = img_lab[mask_bool]

    features = {}
    for i, name in enumerate(['R', 'G', 'B']):
        channel = pixels_rgb[:, i].astype(np.float32)
        features[f'RGB_{name}_mean'] = float(np.mean(channel))
        features[f'RGB_{name}_var'] = float(np.var(channel))
        features[f'RGB_{name}_skew'] = _safe_skew(channel)

    for i, name in enumerate(['L', 'A', 'B']):
        channel = pixels_lab[:, i].astype(np.float32)
        features[f'LAB_{name}_mean'] = float(np.mean(channel))
        features[f'LAB_{name}_var'] = float(np.var(channel))
        features[f'LAB_{name}_skew'] = _safe_skew(channel)

    return features


def compute_texture_features(gray, mask):
    """计算纹理特征：LBP 10-bin 直方图 + GLCM 3个描述符（共13个）"""
    base_features = {
        **{f'LBP_bin{i}': 0.0 for i in range(10)},
        'GLCM_contrast': 0.0,
        'GLCM_energy': 0.0,
        'GLCM_correlation': 0.0,
    }

    if np.sum(mask) == 0:
        return base_features

    # 掩膜约束灰度图：背景置0，仅在前景区域统计 LBP 直方图
    masked_gray = gray.copy()
    masked_gray[mask == 0] = 0

    # LBP (P=8, R=1, uniform)
    lbp = local_binary_pattern(masked_gray, P=8, R=1, method='uniform')
    lbp_vals = lbp[mask > 0].astype(np.uint8)
    hist, _ = np.histogram(lbp_vals, bins=np.arange(0, 11), density=True)
    for i in range(10):
        base_features[f'LBP_bin{i}'] = float(hist[i]) if i < len(hist) else 0.0

    # GLCM：量化到16灰度级，距离1，四个方向取均值
    img_quant = np.uint8(masked_gray / 16)
    glcm = graycomatrix(
        img_quant,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True
    )
    base_features['GLCM_contrast'] = float(np.mean(graycoprops(glcm, 'contrast')))
    base_features['GLCM_energy'] = float(np.mean(graycoprops(glcm, 'energy')))
    corr = np.mean(graycoprops(glcm, 'correlation'))
    base_features['GLCM_correlation'] = float(0.0 if np.isnan(corr) else corr)

    return base_features


def extract_features_for_image(img_path, mask_path):
    """对单张图像提取 35 个特征"""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"无法读取图像：{img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩膜：{mask_path}")
    mask = (mask > 127).astype(np.uint8)

    if mask.shape[:2] != img_rgb.shape[:2]:
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    geo = compute_geometry(mask)
    color = compute_color_features(img_rgb, mask)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    texture = compute_texture_features(gray, mask)

    features = {**geo, **color, **texture}
    if len(features) != 35:
        raise ValueError(f"特征数量错误，当前为 {len(features)}，应为 35")
    return features


def main():
    img_dir = Path(IMG_DIR)
    mask_dir = Path(MASK_DIR)

    image_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    if not image_files:
        print("未找到图像文件，请检查路径")
        return

    records = []
    for img_path in image_files:
        mask_path = mask_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            print(f"跳过 {img_path.name}：无对应掩膜")
            continue
        try:
            features = extract_features_for_image(img_path, mask_path)
            features['filename'] = img_path.name
            records.append(features)
            print(f"已处理：{img_path.name}")
        except Exception as e:
            print(f"处理 {img_path.name} 时出错：{e}")

    if records:
        df = pd.DataFrame(records)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"特征提取完成，共 {len(records)} 张图片，保存至 {OUTPUT_CSV}")
    else:
        print("未提取到任何特征")


if __name__ == "__main__":
    main()
