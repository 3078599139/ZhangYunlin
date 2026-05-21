import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')

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

# ===================== 与室内保持一致的基础预处理 =====================
def resize_and_pad(image, target_size=TARGET_SIZE):
    """等比例缩放并居中填充到目标尺寸。"""
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


def clahe_lab(image_bgr, clip_limit=CLAHE_CLIP, tile_grid_size=CLAHE_GRID):
    """LAB 空间 CLAHE 增强。"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def gaussian_blur(image_bgr, kernel=GAUSSIAN_KERNEL, sigma=GAUSSIAN_SIGMA):
    return cv2.GaussianBlur(image_bgr, kernel, sigma)


# ===================== 野外专用：Reinhard 颜色迁移 =====================
def compute_lab_reference_stats(indoor_preprocess_dir):
    """
    从室内预处理图像计算 LAB 全局均值和标准差，
    用于野外图像的 Reinhard 颜色迁移。
    """
    indoor_preprocess_dir = Path(indoor_preprocess_dir)
    image_files = sorted(f for f in indoor_preprocess_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        raise FileNotFoundError(f"在 {indoor_preprocess_dir} 中未找到室内预处理图像。")

    lab_pixels = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"⚠️ 跳过 {img_file.name}：图像读取失败")
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        lab_pixels.append(lab)

    if not lab_pixels:
        raise ValueError("未能成功读取任何室内预处理图像，无法计算参考统计量。")

    lab_all = np.vstack(lab_pixels)
    mean = lab_all.mean(axis=0)
    std = lab_all.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def reinhard_color_transfer(image_bgr, ref_mean, ref_std):
    """将单张野外图像的 LAB 分布迁移到室内参考分布。"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_mean = lab.reshape(-1, 3).mean(axis=0)
    img_std = lab.reshape(-1, 3).std(axis=0)
    img_std[img_std < 1e-6] = 1.0

    lab_transfer = (lab - img_mean) / img_std * ref_std + ref_mean
    lab_transfer = np.clip(lab_transfer, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_transfer, cv2.COLOR_LAB2BGR)


# ===================== 特征提取 =====================
def preprocess_field_image(img_bgr, ref_mean=None, ref_std=None, use_color_transfer=True):
    """
    野外图像预处理：
    人工边框裁剪后输入 -> 等比例缩放与填充 -> CLAHE -> 高斯滤波 -> Reinhard 颜色迁移（可选）
    返回：RGB 图、LAB 图、梯度幅值图、预处理后的 BGR 图
    """
    if img_bgr is None:
        raise ValueError("输入图像为空。")

    img_bgr = resize_and_pad(img_bgr)
    img_bgr = clahe_lab(img_bgr)
    img_bgr = gaussian_blur(img_bgr)

    if use_color_transfer:
        if ref_mean is None or ref_std is None:
            raise ValueError("启用颜色迁移时，必须提供室内参考均值和标准差。")
        img_bgr = reinhard_color_transfer(img_bgr, ref_mean, ref_std)

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

    return img_rgb, lab, grad_mag, img_bgr


def extract_pixel_features(img_rgb, lab, grad_mag):
    """提取 13 维像素特征。"""
    h, w = img_rgb.shape[:2]

    r, g, b = cv2.split(img_rgb)
    l, a_lab, b_lab = cv2.split(lab)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_hsv, s_hsv, v_hsv = cv2.split(hsv)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    lbp = np.uint8(np.clip(lbp * (255.0 / (LBP_POINTS + 1)), 0, 255))

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


# ===================== 数据组织 =====================
def find_matching_pairs(image_dir, mask_dir, suffix="_mask", mask_ext=".png"):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)

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


def build_field_training_data(
    img_paths,
    mask_paths,
    ref_mean,
    ref_std,
    sample_ratio=0.1,
    random_state=42,
    use_color_transfer=True
):
    """构建野外微调/域适配所需训练数据。"""
    rng = np.random.default_rng(random_state)
    X_list, y_list = [], []

    for img_path, mask_path in zip(img_paths, mask_paths):
        print(f"处理野外标注样本：{Path(img_path).name}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️ 跳过 {img_path}：图像读取失败")
            continue
        img_rgb, lab, grad_mag, _ = preprocess_field_image(
            img_bgr,
            ref_mean=ref_mean,
            ref_std=ref_std,
            use_color_transfer=use_color_transfer
        )
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
        raise ValueError("未能成功构建野外训练数据，请检查图像和掩码。")

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    print(f"野外训练数据构建完成：X {X.shape}, y {y.shape}")
    return X, y


# ===================== 模型 =====================
def load_indoor_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"未找到室内模型：{model_path}")
    clf = joblib.load(model_path)
    print(f"✅ 已加载室内模型：{model_path}")
    return clf


def adapt_rf_with_field_labels(indoor_clf, X_field, y_field, save_path):
    """
    基于野外标注样本进行“域适配”。
    注意：sklearn 的 RandomForestClassifier 不支持真正的增量微调，
    因此这里采用“继承室内模型超参数 + 用野外标注样本重建模型”的方式实现可复现的适配流程。
    """
    params = indoor_clf.get_params()
    # 确保可复现与训练稳定
    params['random_state'] = 42
    params['n_jobs'] = -1

    field_clf = RandomForestClassifier(**params)
    field_clf.fit(X_field, y_field)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(field_clf, save_path)
    print(f"✅ 野外适配模型已保存：{save_path}")
    print(f"   训练集准确率：{field_clf.score(X_field, y_field):.4f}")
    return field_clf


# ===================== 分割与输出 =====================
def segment_field_image(
    img_bgr,
    classifier,
    ref_mean,
    ref_std,
    threshold=0.85,
    use_color_transfer=True
):
    img_rgb, lab, grad_mag, img_bgr_processed = preprocess_field_image(
        img_bgr,
        ref_mean=ref_mean,
        ref_std=ref_std,
        use_color_transfer=use_color_transfer
    )
    features = extract_pixel_features(img_rgb, lab, grad_mag)
    feat_flat = features.reshape(-1, features.shape[-1])
    proba = classifier.predict_proba(feat_flat)[:, 1]
    pred_mask = (proba > threshold).astype(np.uint8).reshape(features.shape[:2]) * 255
    return pred_mask, img_rgb, img_bgr_processed


def refine_mask(mask, min_area=50, max_area=80000):
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


def batch_process_field(
    input_dir,
    output_dir,
    classifier,
    ref_mean,
    ref_std,
    prob_threshold=0.85,
    min_area=50,
    max_area=80000,
    use_color_transfer=True
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到野外图像。")

    for img_file in image_files:
        print(f"处理：{img_file.name}")
        img_bgr = cv2.imread(str(img_file))
        if img_bgr is None:
            print(f"⚠️ 跳过 {img_file.name}：图像读取失败")
            continue

        mask, img_rgb, img_bgr_processed = segment_field_image(
            img_bgr,
            classifier,
            ref_mean=ref_mean,
            ref_std=ref_std,
            threshold=prob_threshold,
            use_color_transfer=use_color_transfer
        )
        mask_refined = refine_mask(mask, min_area=min_area, max_area=max_area)

        cv2.imwrite(str(output_dir / f"{img_file.stem}_mask.png"), mask_refined)
        cv2.imwrite(str(output_dir / f"{img_file.stem}_preprocessed.jpg"), img_bgr_processed)

        overlay = img_rgb.copy()
        overlay[mask_refined > 0] = (0, 255, 0)
        overlay_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        cv2.imwrite(
            str(output_dir / f"{img_file.stem}_overlay.jpg"),
            cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        )
        print("   ✓ 已保存预处理图、掩码和叠加图")

    print(f"批量处理完成，结果保存在：{output_dir}")


if __name__ == "__main__":
    # ===================== 路径配置 =====================
    FIELD_ORIGIN_DIR = "0-Field_origin"
    FIELD_LABEL_DIR = "1-Field_labels"
    INDOOR_MODEL_PATH = "segmentation_rf.pkl"            # 室内训练得到的随机森林模型
    INDOOR_PREPROCESS_DIR = "1-Indoor_preprocess"        # 室内预处理图像，用于计算颜色迁移参考统计量

    OUTPUT_DIR_DIRECT = "2-Field_Segmentation_direct"
    OUTPUT_DIR_ADAPTED = "3-Field_Segmentation_adapted"
    ADAPTED_MODEL_PATH = "field_segmentation_rf.pkl"

    # ===================== 参数配置 =====================
    USE_COLOR_TRANSFER = True     # 是否启用 Reinhard 颜色迁移
    USE_FIELD_ADAPTATION = True   # 是否使用野外标注样本进行“微调/域适配”
    SAMPLE_RATIO = 0.1            # 每张标注图随机采样 10% 像素
    PROB_THRESHOLD = 0.85
    MIN_AREA = 50
    MAX_AREA = 80000
    MASK_SUFFIX = "_mask"
    MASK_EXT = ".png"

    print("=" * 70)
    print("野外针叶分割系统（基于室内模型 + 野外域适配）")
    print("说明：输入图像已在前期完成人工边框裁剪")
    print("=" * 70)

    # 1. 载入室内模型
    indoor_clf = load_indoor_model(INDOOR_MODEL_PATH)

    # 2. 计算室内参考统计量（用于野外颜色迁移）
    if USE_COLOR_TRANSFER:
        ref_mean, ref_std = compute_lab_reference_stats(INDOOR_PREPROCESS_DIR)
        print("✅ 已计算室内图像 LAB 参考统计量")
    else:
        ref_mean, ref_std = None, None
        print("ℹ️ 未启用颜色迁移")

    # 3. 直接用室内模型分割全部野外图像（可作为基线结果）
    print("\n[步骤1] 直接应用室内模型进行野外分割...")
    batch_process_field(
        FIELD_ORIGIN_DIR,
        OUTPUT_DIR_DIRECT,
        indoor_clf,
        ref_mean=ref_mean,
        ref_std=ref_std,
        prob_threshold=PROB_THRESHOLD,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        use_color_transfer=USE_COLOR_TRANSFER
    )

    # 4. 基于野外标注样本进行域适配（方法中的“微调”）
    if USE_FIELD_ADAPTATION:
        print("\n[步骤2] 基于野外标注样本进行域适配...")
        img_paths, mask_paths = find_matching_pairs(
            image_dir=FIELD_ORIGIN_DIR,
            mask_dir=FIELD_LABEL_DIR,
            suffix=MASK_SUFFIX,
            mask_ext=MASK_EXT
        )
        if len(img_paths) == 0:
            raise FileNotFoundError("未找到任何野外标注样本，请检查掩码命名规则。")
        print(f"✅ 找到 {len(img_paths)} 张野外标注图像。")

        X_field, y_field = build_field_training_data(
            img_paths,
            mask_paths,
            ref_mean=ref_mean,
            ref_std=ref_std,
            sample_ratio=SAMPLE_RATIO,
            random_state=42,
            use_color_transfer=USE_COLOR_TRANSFER
        )

        adapted_clf = adapt_rf_with_field_labels(
            indoor_clf,
            X_field,
            y_field,
            save_path=ADAPTED_MODEL_PATH
        )

        print("\n[步骤3] 使用域适配后的模型分割全部野外图像...")
        batch_process_field(
            FIELD_ORIGIN_DIR,
            OUTPUT_DIR_ADAPTED,
            adapted_clf,
            ref_mean=ref_mean,
            ref_std=ref_std,
            prob_threshold=PROB_THRESHOLD,
            min_area=MIN_AREA,
            max_area=MAX_AREA,
            use_color_transfer=USE_COLOR_TRANSFER
        )

    print("\n✨ 全部流程执行完毕！")
    print(f"   - 直接分割结果：{OUTPUT_DIR_DIRECT}")
    if USE_FIELD_ADAPTATION:
        print(f"   - 域适配模型：{ADAPTED_MODEL_PATH}")
        print(f"   - 域适配后分割结果：{OUTPUT_DIR_ADAPTED}")
