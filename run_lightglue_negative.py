from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
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

# SuperPoint+LightGlue（全局初始化，使用CPU避免显存不足）
extractor = SuperPoint(max_num_keypoints=4096).eval()
matcher = LightGlue(features='superpoint', depth_confidence=0.95, width_confidence=0.95).eval()

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
        "text": ["floating debris", "floating object", "water debris"],
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
            conf=0.7,
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
    try:
        image = load_image(target_image_path)
        feats = extractor.extract(image, resize=4096)
        return image, feats
    except Exception as e:
        print(f"⚠️ 提取目标图像特征失败: {e}")
        raise


def match_and_get_bbox(target_feats, query_config, visualize=False, save_path=None, target_image_path='target2.png'):
    """
    使用 LightGlue 进行特征匹配，返回检测框坐标和质量信息

    Args:
        target_feats: 目标图像特征
        query_config: 查询配置
        visualize: 是否可视化匹配结果
        save_path: 保存路径（可选）
        target_image_path: 目标图像路径

    Returns:
        dict: {'bbox': [x_min, y_min, x_max, y_max], 'inliers': int, 'total_matches': int, 'ratio': float} 或 None
    """
    query_image_path = query_config['imagePath']
    size = query_config.get('size', None)
    min_points = query_config.get('point', 4)
    camera_code = query_config.get('cameraCode', 'unknown')

    image1 = load_image(query_image_path)

    if size:
        feats1 = extractor.extract(image1, resize=size)
    else:
        feats1 = extractor.extract(image1)

    matches01 = matcher({'image0': target_feats, 'image1': feats1})
    target_feats_rbd, feats1_rbd, matches01_rbd = [rbd(x) for x in [target_feats, feats1, matches01]]
    matches = matches01_rbd['matches']

    print(f"    LightGlue 匹配: 找到 {len(matches)} 个匹配点")

    # 可视化匹配结果（无论匹配成功与否）
    if visualize:
        try:
            # 直接使用已提取的特征，避免重复提取导致结果不一致
            kpts0_viz = target_feats_rbd["keypoints"]
            kpts1_viz = feats1_rbd["keypoints"]

            if len(matches) > 0:
                m_kpts0_viz = kpts0_viz[matches[..., 0]]
                m_kpts1_viz = kpts1_viz[matches[..., 1]]

                # 重新加载图像用于可视化显示
                image0_viz = load_image(target_image_path)
                image1_viz = load_image(query_image_path)

                # 使用 LightGlue 原生可视化
                axes = viz2d.plot_images([image0_viz, image1_viz])
                viz2d.plot_matches(m_kpts0_viz, m_kpts1_viz, color="lime", lw=0.2)
                viz2d.add_text(0, f'Stop after {matches01_rbd["stop"]} layers\nMatches: {len(matches)}', fs=20)

                # 绘制关键点 pruning 可视化
                kpc0, kpc1 = viz2d.cm_prune(matches01_rbd["prune0"]), viz2d.cm_prune(matches01_rbd["prune1"])
                fig2, axes2 = plt.subplots(1, 1, figsize=(16, 8))
                viz2d.plot_images([image0_viz, image1_viz])
                viz2d.plot_keypoints([kpts0_viz, kpts1_viz], colors=[kpc0, kpc1], ps=10)
                axes2.set_title(f'Keypoints Pruning Visualization\nTotal Matches: {len(matches)}',
                                fontsize=14, fontweight='bold')

                # 如果匹配点数>=4，计算并绘制映射框
                if len(matches) >= 4:
                    pts0 = m_kpts0_viz.cpu().numpy()
                    pts1 = m_kpts1_viz.cpu().numpy()

                    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)

                    if H is not None:
                        query_img_pil = Image.open(query_image_path)
                        query_width, query_height = query_img_pil.size

                        corners_query = np.float32([
                            [0, 0],
                            [query_width, 0],
                            [query_width, query_height],
                            [0, query_height]
                        ]).reshape(-1, 1, 2)

                        corners_target = cv2.perspectiveTransform(corners_query, H).reshape(4, 2)

                        # 在第二张图上绘制映射框
                        axes[1].plot(np.append(corners_target[:, 0], corners_target[0, 0]),
                                     np.append(corners_target[:, 1], corners_target[0, 1]),
                                     'r-', linewidth=3, label='Mapped Region')
                        axes[1].legend(loc='upper right', fontsize=10)
            else:
                # 没有匹配点时也进行可视化
                print(f"    ⚠️ 无匹配点，但仍进行关键点可视化")
                # 重新加载图像用于可视化
                image0_viz = load_image(target_image_path)
                image1_viz = load_image(query_image_path)
                kpc0, kpc1 = viz2d.cm_prune(matches01_rbd.get("prune0", np.zeros(len(kpts0_viz)))), \
                    viz2d.cm_prune(matches01_rbd.get("prune1", np.zeros(len(kpts1_viz))))
                fig2, axes2 = plt.subplots(1, 1, figsize=(16, 8))
                viz2d.plot_images([image0_viz, image1_viz])
                viz2d.plot_keypoints([kpts0_viz, kpts1_viz], colors=[kpc0, kpc1], ps=10)
                axes2.set_title(
                    f'No Matches Found\nImage0 Keypoints: {len(kpts0_viz)}, Image1 Keypoints: {len(kpts1_viz)}',
                    fontsize=14, fontweight='bold')

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"    匹配可视化已保存到: {save_path}")

            plt.show()

        except Exception as e:
            print(f"    ⚠️ 可视化失败: {e}")
            import traceback
            traceback.print_exc()

    if len(matches) < min_points:
        print(f"    ⚠️ 匹配点不足（需要{min_points}个，实际{len(matches)}个）")
        return None

    kpts0 = target_feats_rbd['keypoints']
    kpts1 = feats1_rbd['keypoints']
    m_kpts0 = kpts0[matches[..., 0]]
    m_kpts1 = kpts1[matches[..., 1]]

    pts0 = m_kpts0.cpu().numpy()
    pts1 = m_kpts1.cpu().numpy()

    # 至少需要4个点才能计算单应性矩阵
    if len(pts0) < 4:
        print(f"    ⚠️ 匹配点数不足4个，无法计算单应性矩阵")
        return None

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

    return {
        'bbox': [x_min, y_min, x_max, y_max],
        'corners': corners_target,
        'inliers': inliers_count,
        'total_matches': total_matches,
        'inliers_ratio': inliers_ratio,
        'homography_matrix': H
    }


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
        bbox_result = match_and_get_bbox(target_feats, camera_config, visualize=False)

        if bbox_result is None:
            print(f"  ⚠️ LightGlue 匹配失败，保持 SAM3 原配置不变")
            continue

        bbox = bbox_result['bbox']

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
    try:
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
    except Exception as e:
        print(f"⚠️ SAM3 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def visualize_detection(target_image_path, results, save_path='detection_result.png'):
    """在目标图像上绘制所有检测到的框"""
    if not results:
        print("没有可可视化的检测结果")
        return

    try:
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
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def match_and_get_bbox_from_crop(target_image_path, detection_bbox, query_config, visualize=False, save_path=None):
    """
    从检测框裁剪区域与查询图像进行 LightGlue 匹配

    Args:
        target_image_path: 目标图像路径
        detection_bbox: SAM3 检测框 [x_min, y_min, x_max, y_max]
        query_config: 查询配置
        visualize: 是否可视化
        save_path: 保存路径

    Returns:
        dict: 匹配结果或 None
    """
    import tempfile
    import os
    from datetime import datetime

    query_image_path = query_config['imagePath']
    size = query_config.get('size', None)
    min_points = query_config.get('point', 4)
    camera_code = query_config.get('cameraCode', 'unknown')

    # 读取目标图像并裁剪检测框区域
    target_img_cv = cv2.imread(target_image_path)
    if target_img_cv is None:
        print(f"    ⚠️ 无法读取目标图像: {target_image_path}")
        return None

    x_min, y_min, x_max, y_max = detection_bbox

    # 确保坐标在图像范围内
    h, w = target_img_cv.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        print(f"    ⚠️ 检测框无效")
        return None

    # 裁剪检测框区域
    cropped_img = target_img_cv[y_min:y_max, x_min:x_max]

    if cropped_img.size == 0:
        print(f"    ⚠️ 裁剪区域为空")
        return None

    # 创建保存目录: crop_object/{cameraCode}/
    save_dir = f'./crop_object/{camera_code}'
    os.makedirs(save_dir, exist_ok=True)

    # 按时间格式生成文件名
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # saved_crop_path = f'{save_dir}/crop_{timestamp}.png'
    # cv2.imwrite(saved_crop_path, cropped_img)
    # print(f"    ✓ 裁剪图像已保存: {saved_crop_path}")

    # 将裁剪图像转换为PIL Image用于特征提取（避免磁盘I/O）
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)

    try:
        # 使用 load_image 加载裁剪后的图像（需要临时文件）
        import tempfile
        import os
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        try:
            cv2.imwrite(temp_path, cropped_img)
            image0 = load_image(temp_path)

            # 提取特征
            if size:
                feats0 = extractor.extract(image0, resize=size)
            else:
                feats0 = extractor.extract(image0)
        finally:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # 加载查询图像
        image1_pil = load_image(query_image_path)
        if size:
            feats1 = extractor.extract(image1_pil, resize=size)
        else:
            feats1 = extractor.extract(image1_pil)

        # 匹配
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0_rbd, feats1_rbd, matches01_rbd = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01_rbd['matches']

        print(f"    LightGlue 匹配 (裁剪区域 vs 查询): 找到 {len(matches)} 个匹配点")

        # 可视化（无论匹配成功与否）
        if visualize:
            try:
                kpts0_viz = feats0_rbd["keypoints"]
                kpts1_viz = feats1_rbd["keypoints"]

                if len(matches) > 0:
                    m_kpts0_viz = kpts0_viz[matches[..., 0]]
                    m_kpts1_viz = kpts1_viz[matches[..., 1]]

                    # # 使用 LightGlue 原生可视化
                    axes = viz2d.plot_images([image0, image1_pil])

                    if axes is not None and len(axes) >= 2:
                        viz2d.plot_matches(m_kpts0_viz, m_kpts1_viz, color="lime", lw=0.2)
                        viz2d.add_text(0, f'Stop after {matches01_rbd["stop"]} layers\nMatches: {len(matches)}', fs=20)

                        # 关键点 pruning 可视化
                        kpc0, kpc1 = viz2d.cm_prune(matches01_rbd["prune0"]), viz2d.cm_prune(matches01_rbd["prune1"])
                        fig2, axes2 = plt.subplots(1, 1, figsize=(16, 8))
                        viz2d.plot_images([image0, image1_pil])
                        viz2d.plot_keypoints([kpts0_viz, kpts1_viz], colors=[kpc0, kpc1], ps=10)
                        axes2.set_title(f'Keypoints Pruning\nMatches: {len(matches)}', fontsize=14, fontweight='bold')

                        # 如果匹配点数>=4，绘制映射框
                        if len(matches) >= 4:
                            pts0 = m_kpts0_viz.cpu().numpy()
                            pts1 = m_kpts1_viz.cpu().numpy()

                            H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)

                            if H is not None:
                                query_img_pil = Image.open(query_image_path)
                                query_width, query_height = query_img_pil.size

                                corners_query = np.float32([
                                    [0, 0],
                                    [query_width, 0],
                                    [query_width, query_height],
                                    [0, query_height]
                                ]).reshape(-1, 1, 2)

                                corners_target = cv2.perspectiveTransform(corners_query, H).reshape(4, 2)

                                # 在第二张图上绘制映射框
                                axes[1].plot(np.append(corners_target[:, 0], corners_target[0, 0]),
                                             np.append(corners_target[:, 1], corners_target[0, 1]),
                                             'r-', linewidth=3, label='Mapped Region')
                                axes[1].legend(loc='upper right', fontsize=10)
                            else:
                                print(f"    ⚠️ 可视化: findHomography 返回 None，跳过绘制映射框")
                    else:
                        print(f"    ⚠️ 可视化: plot_images 返回异常，跳过匹配连线")
                else:
                    # 没有匹配点时也进行可视化，显示关键点
                    print(f"    ⚠️ 无匹配点，但仍进行关键点可视化")
                    kpc0, kpc1 = viz2d.cm_prune(matches01_rbd.get("prune0", np.zeros(len(kpts0_viz)))), \
                        viz2d.cm_prune(matches01_rbd.get("prune1", np.zeros(len(kpts1_viz))))
                    fig2, axes2 = plt.subplots(1, 1, figsize=(16, 8))
                    viz2d.plot_images([image0, image1_pil])
                    viz2d.plot_keypoints([kpts0_viz, kpts1_viz], colors=[kpc0, kpc1], ps=10)
                    axes2.set_title(
                        f'No Matches Found\nImage0 Keypoints: {len(kpts0_viz)}, Image1 Keypoints: {len(kpts1_viz)}',
                        fontsize=14, fontweight='bold')

                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"    匹配可视化已保存到: {save_path}")

                plt.show()

            except Exception as e:
                print(f"    ⚠️ 可视化失败: {e}")
                import traceback
                traceback.print_exc()

        if len(matches) < min_points:
            print(f"    ⚠️ 匹配点不足（需要{min_points}个，实际{len(matches)}个）")
            return None

        kpts0 = feats0_rbd['keypoints']
        kpts1 = feats1_rbd['keypoints']
        m_kpts0 = kpts0[matches[..., 0]]
        m_kpts1 = kpts1[matches[..., 1]]

        pts0 = m_kpts0.cpu().numpy()
        pts1 = m_kpts1.cpu().numpy()

        # 至少需要4个点才能计算单应性矩阵
        if len(pts0) < 4:
            print(f"    ⚠️ 匹配点数不足4个，无法计算单应性矩阵")
            return None

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

        # 将映射后的角点转换回原图坐标系
        corners_original = corners_target.copy()
        corners_original[:, 0] += x_min
        corners_original[:, 1] += y_min

        x_coords = corners_original[:, 0]
        y_coords = corners_original[:, 1]
        x_min_orig, y_min_orig = int(x_coords.min()), int(y_coords.min())
        x_max_orig, y_max_orig = int(x_coords.max()), int(y_coords.max())

        inliers_count = int(mask.sum())
        total_matches = len(matches)
        inliers_ratio = inliers_count / total_matches if total_matches > 0 else 0

        print(f"    ✓ 匹配成功 - 边界框: [{x_min_orig}, {y_min_orig}, {x_max_orig}, {y_max_orig}]")
        print(f"    质量: {inliers_count}/{total_matches} (比例: {inliers_ratio:.2f})")

        return {
            'bbox': [x_min_orig, y_min_orig, x_max_orig, y_max_orig],
            'corners': corners_original,
            'inliers': inliers_count,
            'total_matches': total_matches,
            'inliers_ratio': inliers_ratio,
            'homography_matrix': H
        }
    except Exception as e:
        print(f"    ⚠️ 匹配过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_detection_with_negative_samples(detection_bbox, target_image_path, camera_code, negative_configs):
    """
    使用负样本配置验证检测框是否为误检
    将 SAM3 检测框裁剪区域与负样本配置图像进行匹配

    Args:
        detection_bbox: SAM3 检测到的 bbox [x_min, y_min, x_max, y_max]
        target_image_path: 目标图像路径
        camera_code: 相机代码
        negative_configs: label=False 的配置列表

    Returns:
        bool: True 表示是误检（需要过滤），False 表示不是误检
    """
    if not negative_configs:
        return False

    print(f"\n  [误检验证] 检查 {len(negative_configs)} 个负样本配置...")
    print(f"  检测框: {detection_bbox}")

    det_x1, det_y1, det_x2, det_y2 = detection_bbox
    det_center_x = (det_x1 + det_x2) / 2
    det_center_y = (det_y1 + det_y2) / 2
    det_width = det_x2 - det_x1
    det_height = det_y2 - det_y1
    det_area = det_width * det_height

    for neg_idx, neg_config in enumerate(negative_configs):
        neg_class = neg_config.get('class', 'unknown')
        neg_point_threshold = neg_config.get('point', 10)

        print(f"    验证负样本 {neg_idx + 1}: class='{neg_class}', point阈值={neg_point_threshold}")

        # 执行 LightGlue 匹配：裁剪区域 vs 负样本图像
        vis_path = f'neg_match_{neg_idx + 1}.png'
        match_result = match_and_get_bbox_from_crop(
            target_image_path,
            detection_bbox,
            neg_config,
            visualize=False,
            save_path=vis_path
        )

        if match_result is None:
            print(f"      → 匹配失败，跳过")
            continue

        neg_bbox = match_result['bbox']
        neg_inliers = match_result['inliers']
        neg_total = match_result['total_matches']
        neg_ratio = match_result['inliers_ratio']

        # 检查1: 匹配点数是否超过阈值
        if neg_inliers < neg_point_threshold:
            print(f"      → 匹配点数 {neg_inliers} < 阈值 {neg_point_threshold}，跳过")
            continue

        # 检查2: 匹配质量是否良好（内点比例 > 0.3）
        if neg_ratio < 0.3:
            print(f"      → 匹配质量差 (比例 {neg_ratio:.2f} < 0.3)，跳过")
            continue

        # 检查3: 检测框与负样本映射框是否有较大重叠
        neg_x1, neg_y1, neg_x2, neg_y2 = neg_bbox

        # 计算 IoU
        inter_x1 = max(det_x1, neg_x1)
        inter_y1 = max(det_y1, neg_y1)
        inter_x2 = min(det_x2, neg_x2)
        inter_y2 = min(det_y2, neg_y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            neg_area = (neg_x2 - neg_x1) * (neg_y2 - neg_y1)
            union_area = det_area + neg_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            print(f"      → 匹配良好: inliers={neg_inliers}, ratio={neg_ratio:.2f}, IoU={iou:.2f}")

            # 如果 IoU > 0.3，认为是误检
            if iou > 0.3:
                print(f"      ⚠️ 判定为误检！(IoU={iou:.2f} > 0.3)")
                return True
        else:
            # 没有重叠，但匹配质量好，检查中心点距离
            neg_center_x = (neg_x1 + neg_x2) / 2
            neg_center_y = (neg_y1 + neg_y2) / 2
            distance = np.sqrt((det_center_x - neg_center_x) ** 2 + (det_center_y - neg_center_y) ** 2)
            max_distance = max(det_width, det_height) * 1.5

            print(f"      → 无重叠，中心点距离={distance:.0f}, 阈值={max_distance:.0f}")

            if distance < max_distance:
                print(f"      ⚠️ 判定为误检！（距离近且匹配好）")
                return True

    print(f"  ✓ 验证通过，不是误检")
    return False


def filter_false_positives(all_results, target_image_path, camera_code):
    """
    过滤误检测结果

    Args:
        all_results: SAM3 检测结果列表
        target_image_path: 目标图像路径
        camera_code: 相机代码

    Returns:
        list: 过滤后的结果列表
    """
    print("\n" + "=" * 70)
    print("步骤3: 误检过滤")
    print("=" * 70)

    # 获取当前相机的 label=False 配置项
    negative_configs = [
        cam for cam in cameraList
        if cam.get("cameraCode") == camera_code and not cam.get("label", False)
    ]

    if not negative_configs:
        print(f"未找到 cameraCode={camera_code} 且 label=False 的配置，跳过误检过滤")
        return all_results

    print(f"\n找到 {len(negative_configs)} 个负样本配置:")
    for cfg in negative_configs:
        print(f"  - imagePath: {cfg['imagePath']}, class: {cfg.get('class', 'N/A')}, point: {cfg.get('point', 10)}")

    filtered_results = []
    filtered_count = 0

    for idx, result in enumerate(all_results):
        bbox = result['bbox']
        class_name = result['class_name']

        print(f"\n[{idx + 1}/{len(all_results)}] 验证检测框: class='{class_name}', bbox={bbox}")

        is_false_positive = verify_detection_with_negative_samples(
            bbox, target_image_path, camera_code, negative_configs
        )

        if is_false_positive:
            print(f"  ✗ 过滤掉误检框")
            filtered_count += 1
        else:
            print(f"  ✓ 保留检测框")
            filtered_results.append(result)

        # 释放matplotlib资源，防止内存泄漏
        plt.close('all')

    print(f"\n{'=' * 70}")
    print(
        f"误检过滤完成: 原始 {len(all_results)} 个 → 过滤后 {len(filtered_results)} 个 (移除 {filtered_count} 个误检)")
    print(f"{'=' * 70}")

    return filtered_results


if __name__ == "__main__":
    target_image_path = 'target2.png'
    TARGET_CAMERA_CODE = "002"

    # 步骤1: 提取目标图像特征（只执行一次）
    print("=" * 70)
    print("初始化: 提取目标图像特征")
    print("=" * 70)
    target_image, target_feats = extract_target_features(target_image_path)
    print("✓ 目标图像特征提取完成")

    # 步骤2: 构建增强后的 SAM3 配置
    enhanced_sam3_config = build_sam3_config_with_lightglue_hints(
        target_image_path,
        target_feats,
        camera_code=TARGET_CAMERA_CODE
    )

    # 步骤3: 执行 SAM3 检测
    all_results = run_sam3_detection(target_image_path, enhanced_sam3_config)

    # 步骤4: 误检过滤
    if all_results:
        filtered_results = filter_false_positives(all_results, target_image_path, TARGET_CAMERA_CODE)
    else:
        filtered_results = []
        print("\n未检测到任何目标，跳过误检过滤")

    # 步骤5: 汇总结果
    if filtered_results:
        print("\n" + "=" * 70)
        print("最终检测结果详情")
        print("=" * 70)
        for idx, result in enumerate(filtered_results, 1):
            bbox = result['bbox']
            print(f"\n{idx}. 类别: {result['class_name']}")
            print(f"   边界框: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print(f"   置信度: {result['confidence']:.2f}")
            if result.get('has_lightglue_hint'):
                print(f"   LightGlue 提示: {result['lightglue_hint_bbox']}")

        # 步骤6: 可视化
        print("\n" + "=" * 70)
        print("生成可视化结果")
        print("=" * 70)
        visualize_detection(target_image_path, filtered_results, save_path='detection_result.png')
    else:
        print("\n过滤后未检测到任何目标")