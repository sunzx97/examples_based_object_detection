from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from config.config import cameraList

try:
    from ultralytics.models.sam import SAM3SemanticPredictor

    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("警告: 未安装 ultralytics，SAM3 功能不可用")

# SuperPoint+LightGlue（全局初始化）
extractor = SuperPoint(max_num_keypoints=4096).eval().cuda()
matcher = LightGlue(features='superpoint', depth_confidence=0.95, width_confidence=0.95).eval().cuda()

# SAM3 预测器（全局初始化）
sam3_predictor = None

# SAM3 原始检测配置
SAM3_DETECTION_CONFIG = [
    {
        "text": ["person"],
        "bboxes": np.array([[117, 707, 163, 819]]),
        "labels": [0]
    },
    {
        "text": ["bus"],
        "bboxes": None,
        "labels": None
    },
    {
        "text": ["glasses"],
        "bboxes": None,
        "labels": None
    },
    {
        "text": ["collapse"],
        "bboxes": None,
        "labels": None
    }
]


def init_sam3_predictor(model_path="/home/sun/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"):
    """初始化 SAM3 预测器"""
    global sam3_predictor
    if not SAM3_AVAILABLE:
        return None

    if sam3_predictor is None:
        overrides = dict(
            conf=0.5,
            task="segment",
            mode="predict",
            model=model_path,
            half=True,
            save=False,
        )
        sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM3 预测器初始化完成")

    return sam3_predictor


def extract_target_features(target_image_path):
    """提取目标图像的特征（只执行一次）"""
    image = load_image(target_image_path).cuda()
    feats = extractor.extract(image, resize=4096)
    return image, feats


def match_and_get_bbox(target_feats, query_config):
    """
    使用 LightGlue 进行特征匹配，返回检测框坐标

    Returns:
        list: [x_min, y_min, x_max, y_max] 或 None
    """
    query_image_path = query_config['imagePath']
    size = query_config.get('size', None)
    min_points = query_config.get('point', 4)
    camera_code = query_config.get('cameraCode', 'unknown')

    image1 = load_image(query_image_path).cuda()

    if size:
        feats1 = extractor.extract(image1, resize=size)
    else:
        feats1 = extractor.extract(image1)

    matches01 = matcher({'image0': target_feats, 'image1': feats1})
    target_feats_rbd, feats1_rbd, matches01_rbd = [rbd(x) for x in [target_feats, feats1, matches01]]
    matches = matches01_rbd['matches']

    print(f"    LightGlue 匹配: 找到 {len(matches)} 个匹配点")

    if len(matches) < min_points:
        print(f"    ⚠️ 匹配点不足（需要{min_points}个，实际{len(matches)}个）")
        return None

    kpts0 = target_feats_rbd['keypoints']
    kpts1 = feats1_rbd['keypoints']
    m_kpts0 = kpts0[matches[..., 0]]
    m_kpts1 = kpts1[matches[..., 1]]

    pts0 = m_kpts0.cpu().numpy()
    pts1 = m_kpts1.cpu().numpy()

    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)

    if H is None:
        print("    ⚠️ 无法计算单应性矩阵")
        return None

    query_img_pil = Image.open(query_image_path)
    query_width, query_height = query_img_pil.size

    corners_query = np.float32([
        [0, 0],
        [query_width, 0],
        [query_width, query_height],
        [0, query_height]
    ]).reshape(-1, 1, 2)

    corners_target = cv2.perspectiveTransform(corners_query, H).reshape(4, 2)

    x_coords = corners_target[:, 0]
    y_coords = corners_target[:, 1]
    x_min, y_min = int(x_coords.min()), int(y_coords.min())
    x_max, y_max = int(x_coords.max()), int(y_coords.max())

    inliers_count = int(mask.sum())
    total_matches = len(matches)
    inliers_ratio = inliers_count / total_matches if total_matches > 0 else 0

    print(f"    ✓ 匹配成功 - 边界框: [{x_min}, {y_min}, {x_max}, {y_max}]")
    print(f"    质量: {inliers_count}/{total_matches} (比例: {inliers_ratio:.2f})")

    return [x_min, y_min, x_max, y_max]


def build_sam3_config_with_lightglue_hints(target_image_path, target_feats, camera_code=None):
    """
    根据 SAM3 原始配置和特征点匹配配置，构建增强后的 SAM3 配置

    流程：
    1. 复制 SAM3 原始配置
    2. 遍历特征点匹配配置，查找 class 与 SAM3 text 匹配的项
    3. 用 LightGlue 匹配得到 bbox，更新对应 SAM3 配置的 bboxes 和 labels

    Returns:
        list: 增强后的 SAM3 配置列表
    """
    import copy
    enhanced_config = copy.deepcopy(SAM3_DETECTION_CONFIG)

    print("\n" + "=" * 70)
    print("步骤1: 构建增强 SAM3 配置")
    print("=" * 70)

    # 获取所有 label=True 的特征点匹配配置
    labeled_cameras = [cam for cam in cameraList if cam.get("label", False)]

    # 如果指定了 camera_code，进一步筛选
    if camera_code:
        labeled_cameras = [cam for cam in labeled_cameras if cam.get("cameraCode") == camera_code]
        print(f"\n指定相机代码: {camera_code}")

    if not labeled_cameras:
        print("未找到任何 label=True 的特征点匹配配置")
        return enhanced_config

    print(f"\n找到 {len(labeled_cameras)} 个特征点匹配配置:")
    for cam in labeled_cameras:
        print(f"  - cameraCode: {cam['cameraCode']}, class: {cam.get('class', 'N/A')}")

    # 遍历每个特征点匹配配置
    for camera_config in labeled_cameras:
        class_name = camera_config.get('class', None)
        label_value = camera_config.get('label', False)
        if not class_name:
            continue

        print(f"\n处理特征点配置: cameraCode={camera_config['cameraCode']}, class='{class_name}'")

        # 在 SAM3 配置中查找匹配的 text
        matched_sam3_idx = None
        for idx, sam3_item in enumerate(enhanced_config):
            if class_name in sam3_item['text']:
                matched_sam3_idx = idx
                break

        if matched_sam3_idx is None:
            print(f"  ⚠️ 未在 SAM3 配置中找到类别 '{class_name}'，跳过")
            continue

        print(f"  ✓ 匹配到 SAM3 配置索引 {matched_sam3_idx}: text={enhanced_config[matched_sam3_idx]['text']}")

        # 使用 LightGlue 进行特征匹配，获取 bbox
        print(f"  执行 LightGlue 特征匹配...")
        bbox = match_and_get_bbox(target_feats, camera_config)

        if bbox is None:
            print(f"  ⚠️ LightGlue 匹配失败，保持 SAM3 原配置不变")
            continue

        # 根据 label 值确定标签（True -> 1, False -> 0）
        label_int = 1 if label_value else 0
        # 更新 SAM3 配置：追加 bbox 和 label
        current_bboxes = enhanced_config[matched_sam3_idx]['bboxes']
        current_labels = enhanced_config[matched_sam3_idx]['labels']
        if current_bboxes is None:
            # 如果原来没有 bbox，创建新的数组
            enhanced_config[matched_sam3_idx]['bboxes'] = np.array([bbox])
            enhanced_config[matched_sam3_idx]['labels'] = np.array([label_int])
        else:
            # 如果已有 bbox，追加到数组
            enhanced_config[matched_sam3_idx]['bboxes'] = np.vstack([current_bboxes, np.array([bbox])])
            enhanced_config[matched_sam3_idx]['labels'] = np.append(current_labels, label_int)

        print(f"  ✓ 已更新 SAM3 配置[{matched_sam3_idx}]:")
        print(f"    bboxes: {enhanced_config[matched_sam3_idx]['bboxes']}")
        print(f"    labels: {enhanced_config[matched_sam3_idx]['labels']}")

    print("\n" + "=" * 70)
    print("增强后的 SAM3 配置:")
    print("=" * 70)
    for idx, item in enumerate(enhanced_config):
        has_bbox = item['bboxes'] is not None
        print(f"  [{idx}] text={item['text']}, "
              f"bboxes={'有提示' if has_bbox else '无提示'}, "
              f"bboxes值={item['bboxes'].tolist() if has_bbox else None}")

    return enhanced_config


def run_sam3_detection(target_image_path, sam3_config):
    """
    根据 SAM3 配置执行检测

    Args:
        target_image_path: 目标图像路径
        sam3_config: SAM3 检测配置列表

    Returns:
        list: 所有检测结果
    """
    predictor = init_sam3_predictor()
    if predictor is None:
        print("SAM3 预测器未初始化")
        return []

    print("\n" + "=" * 70)
    print("步骤2: 执行 SAM3 检测")
    print("=" * 70)

    # 设置图像
    predictor.set_image(target_image_path)

    all_results = []

    # 遍历每个配置项
    for idx, config_item in enumerate(sam3_config):
        text = config_item['text']
        bboxes = config_item['bboxes']
        labels = config_item['labels']

        print(f"\n检测类别: {text}")
        if bboxes is not None:
            print(f"  使用 bbox 提示: {bboxes.tolist()}")
        else:
            print(f"  无 bbox 提示")

        # 执行检测
        results = predictor(
            text=text,
            bboxes=bboxes,
            labels=labels
        )

        # 解析结果
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_data = result.boxes.data.cpu().numpy()
                for box in boxes_data:
                    x1, y1, x2, y2, conf, cls_id = box
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': text[0],
                        'has_lightglue_hint': bboxes is not None,
                        'lightglue_hint_bbox': bboxes.tolist() if bboxes is not None else None
                    }
                    all_results.append(detection)
                    print(f"  ✓ 检测到: bbox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], "
                          f"置信度={conf:.2f}")

    print(f"\n总共检测到 {len(all_results)} 个目标")
    return all_results


def visualize_detection(target_image_path, results, save_path='detection_result.png'):
    """在目标图像上绘制所有检测到的框"""
    if not results:
        print("没有可可视化的检测结果")
        return

    img = cv2.imread(target_image_path)
    if img is None:
        print(f"无法读取图像: {target_image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_rgb)

    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(results), 1)))

    for idx, (result, color) in enumerate(zip(results, colors)):
        bbox = result['bbox']
        class_name = result['class_name']
        conf = result['confidence']
        has_hint = result.get('has_lightglue_hint', False)

        # 绘制检测框（实线）
        rect_x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
        rect_y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
        linewidth = 3 if has_hint else 2
        linestyle = '-' if has_hint else '--'
        ax.plot(rect_x, rect_y, linestyle, color=color, linewidth=linewidth,
                label=f'{class_name} ({"w/ hint" if has_hint else "no hint"})')

        # 如果有 LightGlue 提示框，绘制虚线框
        if has_hint and result.get('lightglue_hint_bbox'):
            hint_bboxes = result['lightglue_hint_bbox']

            # 处理单个 bbox 或多个 bbox 的情况
            if isinstance(hint_bboxes[0], list):
                # 多个 bbox 的情况：[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
                for hint_bbox in hint_bboxes:
                    hint_x = [hint_bbox[0], hint_bbox[2], hint_bbox[2], hint_bbox[0], hint_bbox[0]]
                    hint_y = [hint_bbox[1], hint_bbox[1], hint_bbox[3], hint_bbox[3], hint_bbox[1]]
                    ax.plot(hint_x, hint_y, ':', color=color, linewidth=2, alpha=0.5)
            else:
                # 单个 bbox 的情况：[x1,y1,x2,y2]
                hint_x = [hint_bboxes[0], hint_bboxes[2], hint_bboxes[2], hint_bboxes[0], hint_bboxes[0]]
                hint_y = [hint_bboxes[1], hint_bboxes[1], hint_bboxes[3], hint_bboxes[3], hint_bboxes[1]]
                ax.plot(hint_x, hint_y, ':', color=color, linewidth=2, alpha=0.5)

        # 添加标签
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        label_text = f'{class_name}\n{conf:.2f}'
        ax.text(center_x, center_y - 10, label_text,
                fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7))

    ax.set_title(f'SAM3 Detection Results ({len(results)} objects)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('off')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {save_path}")
    plt.show()


if __name__ == "__main__":
    target_image_path = 'target.png'

    # 步骤1: 提取目标图像特征（只执行一次）
    print("=" * 70)
    print("初始化: 提取目标图像特征")
    print("=" * 70)
    target_image, target_feats = extract_target_features(target_image_path)
    print("✓ 目标图像特征提取完成")

    # 步骤2: 构建增强后的 SAM3 配置
    enhanced_sam3_config = build_sam3_config_with_lightglue_hints(target_image_path, target_feats, camera_code="001")

    # 步骤3: 执行 SAM3 检测
    all_results = run_sam3_detection(target_image_path, enhanced_sam3_config)

    # 步骤4: 汇总结果
    if all_results:
        print("\n" + "=" * 70)
        print("检测结果详情")
        print("=" * 70)
        for idx, result in enumerate(all_results, 1):
            bbox = result['bbox']
            print(f"\n{idx}. 类别: {result['class_name']}")
            print(f"   边界框: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print(f"   置信度: {result['confidence']:.2f}")
            if result.get('has_lightglue_hint'):
                print(f"   LightGlue 提示: {result['lightglue_hint_bbox']}")

        # 步骤5: 可视化
        print("\n" + "=" * 70)
        print("生成可视化结果")
        print("=" * 70)
        visualize_detection(target_image_path, all_results, save_path='detection_result.png')
    else:
        print("\n未检测到任何目标")
