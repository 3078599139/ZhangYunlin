import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============== 全局配置 ==============
TARGET_SIZE = (1300, 1300)   # (宽, 高)
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)
GAUSSIAN_KERNEL = (3, 3)
GAUSSIAN_SIGMA = 0.5
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = 'uniform'


# ============== 预处理：与 1.1 保持完全一致 ==============
def resize_and_pad(image, target_size=TARGET_SIZE):
    """缩放并居中填充至目标尺寸，保持比例，不足部分用黑色填充。"""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        raise ValueError("输入图像尺寸无效。")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def clahe_lab(image, clip_limit=CLAHE_CLIP, tile_grid_size=CLAHE_GRID):
    """LAB 空间 CLAHE 增强。"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def gaussian_blur(image, kernel=GAUSSIAN_KERNEL, sigma=GAUSSIAN_SIGMA):
    """高斯模糊去噪。"""
    return cv2.GaussianBlur(image, kernel, sigma)


def preprocess_image(img_bgr):
    """
    输入 BGR 图像，输出：
    1) 预处理后的 RGB 图
    2) 预处理后的 LAB 图
    3) Sobel 梯度幅值图
    预处理流程与 1.1 完全一致：等比例缩放填充 -> CLAHE -> 高斯模糊
    """
    if img_bgr is None:
        raise ValueError("输入图像为空。")

    img_bgr = resize_and_pad(img_bgr)
    img_bgr = clahe_lab(img_bgr)
    img_bgr = gaussian_blur(img_bgr)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    if grad_mag.max() > 0:
        grad_mag = np.uint8(np.clip(grad_mag / grad_mag.max() * 255, 0, 255))
    else:
        grad_mag = np.zeros_like(gray, dtype=np.uint8)

    return img_rgb, lab, grad_mag


def extract_pixel_features(img_rgb, lab, grad_mag):
    """提取 13 维像素特征。"""
    h, w = img_rgb.shape[:2]

    # 颜色特征
    r, g, b = cv2.split(img_rgb)
    l, a_lab, b_lab = cv2.split(lab)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_hsv, s_hsv, v_hsv = cv2.split(hsv)

    # 纹理特征：逐像素 uniform-LBP 编码值
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    lbp = np.uint8(np.clip(lbp * (255.0 / (LBP_POINTS + 1)), 0, 255))

    # 空间坐标（归一化）
    x_coords = np.tile(np.arange(w), (h, 1)).astype(np.float32) / w
    y_coords = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float32) / h

    features = np.dstack([
        r, g, b,
        l, a_lab, b_lab,
        h_hsv, s_hsv, v_hsv,
        grad_mag,
        lbp,
        x_coords, y_coords
    ])
    return features


def mask_resize_and_pad(mask, target_size=TARGET_SIZE):
    """对单通道掩码执行与图像一致的等比例缩放和黑边填充。"""
    h, w = mask.shape[:2]
    target_w, target_h = target_size

    if h == 0 or w == 0:
        raise ValueError("输入掩码尺寸无效。")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def find_matching_pairs(image_dir, mask_dir, suffix="_mask", mask_ext=".png"):
    """匹配原始图像与标注掩码（自动过滤无掩码的图片）。"""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in image_exts)

    img_paths, mask_paths = [], []
    for img_file in image_files:
        mask_name = f"{img_file.stem}{suffix}{mask_ext}"
        mask_file = mask_dir / mask_name
        if mask_file.exists():
            img_paths.append(str(img_file))
            mask_paths.append(str(mask_file))
        else:
            print(f"⚠️ 跳过 {img_file.name}：无对应掩码")
    return img_paths, mask_paths


def build_training_data(img_paths, mask_paths, sample_ratio=0.1, random_state=42):
    """构建训练集（图像和掩码均统一为 TARGET_SIZE）。"""
    rng = np.random.default_rng(random_state)
    X_list, y_list = [], []

    for img_path, mask_path in zip(img_paths, mask_paths):
        print(f"处理训练样本：{Path(img_path).name}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️ 跳过 {img_path}：图像读取失败")
            continue
        img_rgb, lab, grad_mag = preprocess_image(img_bgr)
        features = extract_pixel_features(img_rgb, lab, grad_mag)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ 跳过 {mask_path}：掩码读取失败")
            continue
        mask = mask_resize_and_pad(mask, target_size=TARGET_SIZE)
        mask_bin = (mask > 127).astype(np.uint8).flatten()

        feat_flat = features.reshape(-1, features.shape[-1])

        n_pixels = len(mask_bin)
        n_sample = max(1, int(n_pixels * sample_ratio))
        indices = rng.choice(n_pixels, n_sample, replace=False)

        X_list.append(feat_flat[indices])
        y_list.append(mask_bin[indices])

    if not X_list:
        raise ValueError("未能成功构建训练数据，请检查图像和掩码。")

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    print(f"训练数据构建完成：X {X.shape}, y {y.shape}")
    return X, y


def train_segmentation_model(X, y, save_path="models/segmentation_rf.pkl"):
    """训练随机森林分类器。"""
    print("开始训练随机森林...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        max_samples=0.7,
        n_jobs=-1,
        random_state=42,
        class_weight={0: 1, 1: 2}
    )
    clf.fit(X, y)
    print(f"训练完成，训练集准确率：{clf.score(X, y):.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"模型已保存：{save_path}")
    return clf


def segment_image(img_bgr, classifier, threshold=0.85):
    """使用概率阈值分割，返回二值掩码和预处理 RGB 图。"""
    img_rgb, lab, grad_mag = preprocess_image(img_bgr)
    features = extract_pixel_features(img_rgb, lab, grad_mag)
    feat_flat = features.reshape(-1, features.shape[-1])
    proba = classifier.predict_proba(feat_flat)[:, 1]
    pred_mask = (proba > threshold).astype(np.uint8) * 255
    pred_mask = pred_mask.reshape(features.shape[:2])
    return pred_mask, img_rgb


def refine_mask(mask, min_area=50, max_area=80000):
    """形态学开闭运算并按连通域面积过滤。"""
    if mask.ndim == 1:
        mask = mask.reshape(TARGET_SIZE[1], TARGET_SIZE[0])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.drawContours(mask_filtered, [cnt], -1, 255, thickness=cv2.FILLED)
    return mask_filtered


def batch_process(input_dir, output_dir, classifier, prob_threshold=0.85, min_area=50, max_area=80000):
    """批量分割并保存掩码和叠加图。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in image_exts)

    for img_file in image_files:
        print(f"处理：{img_file.name}")
        img_bgr = cv2.imread(str(img_file))
        if img_bgr is None:
            print(f"⚠️ 跳过 {img_file.name}：图像读取失败")
            continue

        mask, img_rgb = segment_image(img_bgr, classifier, threshold=prob_threshold)
        mask_refined = refine_mask(mask, min_area=min_area, max_area=max_area)

        cv2.imwrite(str(output_dir / f"{img_file.stem}_mask.png"), mask_refined)

        overlay = img_rgb.copy()
        overlay[mask_refined > 0] = (0, 255, 0)
        overlay_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        cv2.imwrite(
            str(output_dir / f"{img_file.stem}_overlay.jpg"),
            cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        )
        print("   ✓ 已保存掩码和叠加图")

    print(f"批量处理完成，结果保存在：{output_dir}")


if __name__ == "__main__":
    ORIGIN_DIR = "0-Indoor_origin"
    LABEL_DIR = "1-Indoor_labels"
    OUTPUT_DIR = "2-Indoor_Segmentation"
    MODEL_PATH = "segmentation_rf.pkl"

    print("=" * 60)
    print("针叶分割系统（等比例缩放填充 + CLAHE + 高斯去噪）")
    print("说明：输入图像已在前期完成人工边框裁剪")
    print("=" * 60)

    img_paths, mask_paths = find_matching_pairs(
        image_dir=ORIGIN_DIR,
        mask_dir=LABEL_DIR,
        suffix="_mask",
        mask_ext=".png"
    )
    if len(img_paths) == 0:
        raise FileNotFoundError("未找到任何标注数据！请检查掩码命名规则。")
    print(f"✅ 找到 {len(img_paths)} 张标注图像。")

    X_train, y_train = build_training_data(img_paths, mask_paths, sample_ratio=0.1, random_state=42)
    clf = train_segmentation_model(X_train, y_train, save_path=MODEL_PATH)

    batch_process(
        ORIGIN_DIR,
        OUTPUT_DIR,
        clf,
        prob_threshold=0.85,
        min_area=50,
        max_area=80000
    )

    print("\n✨ 全部流程执行完毕！")
    print(f"   - 分割结果：{OUTPUT_DIR}")
    print(f"   - 训练模型：{MODEL_PATH}")
