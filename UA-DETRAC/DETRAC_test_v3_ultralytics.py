import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("警告: 未安装 ultralytics，SAM3 功能不可用")


class DETRACTester:
    def __init__(self, model_path, device='cuda', confidence_threshold=0.5, iou_threshold=0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        print("加载 SAM3 模型...")
        if not SAM3_AVAILABLE:
            raise ImportError("SAM3SemanticPredictor 不可用，请安装 ultralytics")
        
        # 使用 ultralytics 的 SAM3SemanticPredictor，它自动处理 dtype 问题
        overrides = dict(
            conf=confidence_threshold,
            task="segment",
            mode="predict",
            model=model_path,
            half=False,  # 使用 Float32 避免混合精度问题
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM3 预测器初始化完成!")

        self.text_prompt = "car or vehicle"

        # 初始化 LightGlue 相关组件
        self._init_lightglue()

        # 加载配置文件
        self._load_camera_config()

    def _init_lightglue(self):
        """初始化 LightGlue 特征提取器和匹配器"""
        try:
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import load_image, rbd

            print("初始化 LightGlue...")
            self.extractor = SuperPoint(max_num_keypoints=4096).eval().cuda()
            self.matcher = LightGlue(features='superpoint', depth_confidence=0.95, width_confidence=0.95).eval().cuda()
            self.load_image = load_image
            self.rbd = rbd
            self.lightglue_available = True
            print("LightGlue 初始化完成")
        except ImportError as e:
            print("警告: 未安装 lightglue，将不使用特征匹配功能:", e)
            self.lightglue_available = False

    def _load_camera_config(self):
        """加载相机配置文件"""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from config.config import cameraList
            self.camera_list = cameraList

            # 按 cameraCode 分组配置
            self.camera_config_map = {}
            for cam in cameraList:
                code = cam.get('cameraCode')
                if code:
                    if code not in self.camera_config_map:
                        self.camera_config_map[code] = []
                    self.camera_config_map[code].append(cam)

            print(f"加载了 {len(self.camera_config_map)} 个相机的配置")
        except Exception as e:
            print(f"警告: 加载配置文件失败 - {e}")
            self.camera_config_map = {}

    def extract_target_features(self, target_image_path):
        """提取目标图像的特征"""
        if not self.lightglue_available:
            return None

        try:
            image = self.load_image(target_image_path).cuda()
            feats = self.extractor.extract(image, resize=4096)
            return image, feats
        except Exception as e:
            print(f"  提取目标图像特征失败: {e}")
            return None

    def extract_current_frame_features(self, frame_path):
        """提取当前帧的特征"""
        if not self.lightglue_available:
            return None
        
        try:
            image = self.load_image(frame_path).cuda()
            feats = self.extractor.extract(image, resize=4096)
            return feats
        except Exception as e:
            print(f"  提取当前帧特征失败: {e}")
            return None

    def match_and_get_bbox(self, target_feats, query_config):
        """
        使用 LightGlue 进行特征匹配，返回检测框坐标

        Returns:
            list: [x_min, y_min, x_max, y_max] 或 None
        """
        if not self.lightglue_available:
            return None

        import cv2

        query_image_path = query_config['imagePath']
        size = query_config.get('size', None)
        min_points = query_config.get('point', 4)

        try:
            image1 = self.load_image(query_image_path).cuda()

            if size:
                feats1 = self.extractor.extract(image1, resize=size)
            else:
                feats1 = self.extractor.extract(image1)

            matches01 = self.matcher({'image0': target_feats, 'image1': feats1})
            target_feats_rbd, feats1_rbd, matches01_rbd = [self.rbd(x) for x in [target_feats, feats1, matches01]]
            matches = matches01_rbd['matches']

            if len(matches) < min_points:
                return None

            kpts0 = target_feats_rbd['keypoints']
            kpts1 = feats1_rbd['keypoints']
            m_kpts0 = kpts0[matches[..., 0]]
            m_kpts1 = kpts1[matches[..., 1]]

            pts0 = m_kpts0.cpu().numpy()
            pts1 = m_kpts1.cpu().numpy()

            H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)

            if H is None:
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

            return [x_min, y_min, x_max, y_max]
        except Exception as e:
            print(f"  LightGlue 匹配失败: {e}")
            return None

    def compute_iou(self, box1, box2):
        """
        计算两个边界框的 IoU
        box format: [left, top, width, height]
        """
        x1_min, y1_min = box1[0], box1[1]
        x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

        x2_min, y2_min = box2[0], box2[1]
        x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def load_gt_annotations(self, gt_xml_path):
        """
        加载 GT XML 标注

        Returns:
            gt_data: dict, key=frame_num, value=list of targets
            ignored_regions: list of boxes in [left, top, width, height] format
        """
        tree = ET.parse(gt_xml_path)
        root = tree.getroot()

        gt_data = {}
        ignored_regions = []

        # 加载 ignored regions
        ignored_region_elem = root.find('ignored_region')
        if ignored_region_elem is not None:
            for box_elem in ignored_region_elem.findall('box'):
                ignored_box = [
                    float(box_elem.get('left')),
                    float(box_elem.get('top')),
                    float(box_elem.get('width')),
                    float(box_elem.get('height'))
                ]
                ignored_regions.append(ignored_box)

        for frame_elem in root.findall('frame'):
            frame_num = int(frame_elem.get('num'))
            targets = []

            target_list = frame_elem.find('target_list')
            if target_list is not None:
                for target_elem in target_list.findall('target'):
                    target_id = int(target_elem.get('id'))

                    box_elem = target_elem.find('box')
                    box = [
                        float(box_elem.get('left')),
                        float(box_elem.get('top')),
                        float(box_elem.get('width')),
                        float(box_elem.get('height'))
                    ]

                    attr_elem = target_elem.find('attribute')
                    vehicle_type = attr_elem.get('vehicle_type', 'car') if attr_elem is not None else 'car'

                    targets.append({
                        'id': target_id,
                        'box': box,
                        'vehicle_type': vehicle_type,
                    })

            gt_data[frame_num] = targets

        return gt_data, ignored_regions

    def is_box_in_ignored_region(self, pred_box, ignored_regions):
        """
        检查预测框是否在 ignored region 内

        Args:
            pred_box: [left, top, width, height]
            ignored_regions: list of [left, top, width, height]

        Returns:
            bool: True if the prediction box center is inside any ignored region
        """
        pred_center_x = pred_box[0] + pred_box[2] / 2
        pred_center_y = pred_box[1] + pred_box[3] / 2

        for ign_box in ignored_regions:
            ign_left = ign_box[0]
            ign_top = ign_box[1]
            ign_right = ign_box[0] + ign_box[2]
            ign_bottom = ign_box[1] + ign_box[3]

            if (ign_left <= pred_center_x <= ign_right and
                    ign_top <= pred_center_y <= ign_bottom):
                return True

        return False

    def detect_frame(self, image_path, ignored_regions=None, lightglue_configs=None):
        """检测单帧图像
            
        Args:
            image_path: 图像路径
            ignored_regions: 忽略区域列表
            lightglue_configs: LightGlue 配置列表（用于逐帧匹配）
        """
        # 先进行 LightGlue 匹配（如果需要）
        lightglue_bboxes = []
        if lightglue_configs and len(lightglue_configs) > 0:
            try:
                # 提取当前帧的特征
                current_frame_feats = self.extract_current_frame_features(image_path)
                    
                if current_frame_feats is not None:
                    for config in lightglue_configs:
                        bbox = self.match_and_get_bbox(current_frame_feats, config)
                        if bbox is not None:
                            lightglue_bboxes.append(bbox)
            except Exception as e:
                import traceback
                print(f"  LightGlue 匹配失败: {e}")
                print(f"  错误详情: {traceback.format_exc()}")
        
        # 然后进行 SAM3 检测
        # 设置图像
        self.predictor.set_image(image_path)
        
        # 准备 bbox 提示
        bboxes_array = None
        labels_array = None
        if lightglue_bboxes:
            bboxes_array = np.array(lightglue_bboxes, dtype=np.float32)
            labels_array = np.ones(len(lightglue_bboxes), dtype=np.int32)
        
        # 执行检测
        results = self.predictor(
            text=[self.text_prompt],
            bboxes=bboxes_array,
            labels=labels_array
        )
        
        # 解析结果
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_data = result.boxes.data.cpu().numpy()
                for box in boxes_data:
                    x1, y1, x2, y2, conf, cls_id = box
                    
                    # 转换为 [left, top, width, height] 格式
                    left = float(x1)
                    top = float(y1)
                    width = float(x2 - x1)
                    height = float(y2 - y1)
                    pred_box = [left, top, width, height]
                    
                    # 过滤在 ignored region 内的检测
                    if ignored_regions and self.is_box_in_ignored_region(pred_box, ignored_regions):
                        continue
                    
                    # 过滤过小的检测框（面积 < 100 像素²）
                    if width * height < 100:
                        continue
                    
                    # 过滤低置信度
                    if conf < self.confidence_threshold:
                        continue
                    
                    detections.append({
                        'box': [left, top, width, height],
                        'confidence': float(conf),
                        'category': 'car'
                    })
        
        return detections

    def match_predictions_to_gt(self, predictions, gt_targets):
        """
        使用 IoU 将预测结果匹配到 GT

        Returns:
            matched_results: list of dict
            unmatched_gt_ids: list of int (漏检的 GT IDs)
        """
        num_preds = len(predictions)
        num_gts = len(gt_targets)

        pred_matched = [False] * num_preds
        gt_matched = [False] * num_gts

        matched_results = []

        for pred_idx, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_targets):
                if gt_matched[gt_idx]:
                    continue

                iou = self.compute_iou(pred['box'], gt['box'])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                pred_matched[pred_idx] = True
                gt_matched[best_gt_idx] = True

                matched_results.append({
                    'gt_id': gt_targets[best_gt_idx]['id'],
                    'pred_box': pred['box'],
                    'gt_box': gt_targets[best_gt_idx]['box'],
                    'iou': best_iou,
                    'confidence': pred['confidence'],
                    'is_tp': True,
                    'is_fp': False,
                    'vehicle_type': gt_targets[best_gt_idx]['vehicle_type']
                })
            else:
                matched_results.append({
                    'gt_id': None,
                    'pred_box': pred['box'],
                    'gt_box': None,
                    'iou': best_iou,
                    'confidence': pred['confidence'],
                    'is_tp': False,
                    'is_fp': True,
                    'vehicle_type': 'car'
                })

        unmatched_gt_ids = [
            gt_targets[gt_idx]['id']
            for gt_idx in range(num_gts)
            if not gt_matched[gt_idx]
        ]

        return matched_results, unmatched_gt_ids

    def create_xml_for_video(self, video_name, frame_results, output_path):
        """
        创建包含 GT 和检测结果的 XML 文件
        """
        sequence = ET.Element('sequence')
        sequence.set('name', video_name)

        seq_attr = ET.SubElement(sequence, 'sequence_attribute')
        seq_attr.set('camera_state', 'unknown')
        seq_attr.set('sence_weather', 'unknown')

        ignored_region = ET.SubElement(sequence, 'ignored_region')

        for frame_num in sorted(frame_results.keys()):
            result = frame_results[frame_num]

            frame_elem = ET.SubElement(sequence, 'frame')
            frame_elem.set('density', '1')
            frame_elem.set('num', str(frame_num))

            target_list = ET.SubElement(frame_elem, 'target_list')

            match_map = {}
            for match in result['matched_results']:
                if match['is_tp']:
                    match_map[match['gt_id']] = match

            for gt_id in result['all_gt_ids']:
                gt_info = result['gt_id_map'].get(gt_id, {})

                target = ET.SubElement(target_list, 'target')
                target.set('id', str(gt_id))

                if 'box' in gt_info:
                    box = ET.SubElement(target, 'box')
                    box.set('left', f"{gt_info['box'][0]:.2f}")
                    box.set('top', f"{gt_info['box'][1]:.2f}")
                    box.set('width', f"{gt_info['box'][2]:.2f}")
                    box.set('height', f"{gt_info['box'][3]:.2f}")

                attribute = ET.SubElement(target, 'attribute')
                attribute.set('orientation', '0.0')
                attribute.set('speed', '0.0')
                attribute.set('trajectory_length', '1')
                attribute.set('truncation_ratio', '0')
                attribute.set('vehicle_type', gt_info.get('vehicle_type', 'unknown'))

                if gt_id in match_map:
                    match = match_map[gt_id]
                    attribute.set('detection_status', 'TP')
                    attribute.set('iou', f"{match['iou']:.4f}")
                    attribute.set('confidence', f"{match['confidence']:.4f}")
                    pred_box = ET.SubElement(target, 'pred_box')
                    pred_box.set('left', f"{match['pred_box'][0]:.2f}")
                    pred_box.set('top', f"{match['pred_box'][1]:.2f}")
                    pred_box.set('width', f"{match['pred_box'][2]:.2f}")
                    pred_box.set('height', f"{match['pred_box'][3]:.2f}")
                else:
                    attribute.set('detection_status', 'FN')
                    attribute.set('confidence', '0.0')

            for match in result['matched_results']:
                if match['is_fp']:
                    target = ET.SubElement(target_list, 'target')

                    box = ET.SubElement(target, 'box')
                    box.set('left', f"{match['pred_box'][0]:.2f}")
                    box.set('top', f"{match['pred_box'][1]:.2f}")
                    box.set('width', f"{match['pred_box'][2]:.2f}")
                    box.set('height', f"{match['pred_box'][3]:.2f}")

                    attribute = ET.SubElement(target, 'attribute')
                    attribute.set('orientation', '0.0')
                    attribute.set('speed', '0.0')
                    attribute.set('trajectory_length', '0')
                    attribute.set('truncation_ratio', '0')
                    attribute.set('vehicle_type', 'car')
                    attribute.set('detection_status', 'FP')
                    attribute.set('confidence', f"{match['confidence']:.4f}")

        xml_str = ET.tostring(sequence, encoding='utf-8')

        from xml.dom.minidom import parseString
        pretty_xml = parseString(xml_str).toprettyxml(indent="   ", encoding='utf-8')

        with open(output_path, 'wb') as f:
            f.write(pretty_xml)

    def test_dataset(self, images_root, gt_xml_root, output_root):
        """
        测试整个 DETRAC 数据集
        """
        os.makedirs(output_root, exist_ok=True)

        video_folders = sorted(glob.glob(os.path.join(images_root, 'MVI_*')))
        print(f"找到 {len(video_folders)} 个视频文件夹")

        overall_stats = {
            'total_frames': 0,
            'total_gt': 0,
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
            'per_video': {}
        }

        for video_folder in tqdm(video_folders, desc="处理视频"):
            video_name = os.path.basename(video_folder)

            image_files = sorted(glob.glob(os.path.join(video_folder, 'img*.jpg')))
            if not image_files:
                print(f"警告: {video_name} 中没有找到图像")
                continue

            gt_xml_path = os.path.join(gt_xml_root, f"{video_name}.xml")
            if not os.path.exists(gt_xml_path):
                print(f"警告: {video_name} 的 GT XML 不存在，跳过")
                continue

            print(f"\n处理视频: {video_name} ({len(image_files)} 帧)")

            # 获取该视频的 LightGlue 配置
            lightglue_configs = []
                        
            if self.lightglue_available and len(image_files) > 0:
                # 从视频名称提取 cameraCode (MVI_20011 -> 20011)
                parts = video_name.split('_')
                if len(parts) >= 2:
                    camera_code = parts[1]
            
                    if camera_code in self.camera_config_map:
                        configs = self.camera_config_map[camera_code]
                        # 只处理 label=True 的配置
                        lightglue_configs = [cfg for cfg in configs if cfg.get('label', False)]
            
                        if lightglue_configs:
                            print(f"  找到 {len(lightglue_configs)} 个 LightGlue 配置，将逐帧进行匹配")
                        else:
                            print(f"  ℹ 无 LightGlue 配置，使用纯文本提示")
                    else:
                        print(f"  ℹ 未找到 cameraCode={camera_code} 的配置，使用纯文本提示")
                else:
                    print(f"  ℹ 无法解析视频名称，使用纯文本提示")
            else:
                print(f"  ℹ LightGlue 不可用，使用纯文本提示")

            gt_data, ignored_regions = self.load_gt_annotations(gt_xml_path)
            print(f"  找到 {len(ignored_regions)} 个 ignored regions")
            frame_results = {}

            video_stats = {
                'frames': 0,
                'gt_count': 0,
                'tp_count': 0,
                'fp_count': 0,
                'fn_count': 0,
            }

            for img_path in tqdm(image_files, desc=f"  {video_name}", leave=False):
                frame_filename = os.path.basename(img_path)
                frame_num = int(frame_filename.replace('img', '').replace('.jpg', ''))

                gt_targets = gt_data.get(frame_num, [])
                
                try:
                    predictions = self.detect_frame(img_path, ignored_regions, lightglue_configs)

                    matched_results, unmatched_gt_ids = self.match_predictions_to_gt(
                        predictions, gt_targets
                    )

                    gt_id_map = {t['id']: t for t in gt_targets}

                    frame_results[frame_num] = {
                        'matched_results': matched_results,
                        'unmatched_gt_ids': unmatched_gt_ids,
                        'all_gt_ids': [t['id'] for t in gt_targets],
                        'gt_id_map': gt_id_map,
                    }

                    tp = sum(1 for m in matched_results if m['is_tp'])
                    fp = sum(1 for m in matched_results if m['is_fp'])
                    fn = len(unmatched_gt_ids)

                    video_stats['frames'] += 1
                    video_stats['gt_count'] += len(gt_targets)
                    video_stats['tp_count'] += tp
                    video_stats['fp_count'] += fp
                    video_stats['fn_count'] += fn

                except Exception as e:
                    print(f"  错误: 帧 {frame_num} 检测失败 - {str(e)}")
                    frame_results[frame_num] = {
                        'matched_results': [],
                        'unmatched_gt_ids': [t['id'] for t in gt_targets],
                        'all_gt_ids': [t['id'] for t in gt_targets],
                        'gt_id_map': {t['id']: t for t in gt_targets},
                    }

            output_xml = os.path.join(output_root, f"{video_name}.xml")
            self.create_xml_for_video(video_name, frame_results, output_xml)

            precision = video_stats['tp_count'] / (video_stats['tp_count'] + video_stats['fp_count']) if (video_stats[
                                                                                                              'tp_count'] +
                                                                                                          video_stats[
                                                                                                              'fp_count']) > 0 else 0
            recall = video_stats['tp_count'] / (video_stats['tp_count'] + video_stats['fn_count']) if (video_stats[
                                                                                                           'tp_count'] +
                                                                                                       video_stats[
                                                                                                           'fn_count']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            overall_stats['total_frames'] += video_stats['frames']
            overall_stats['total_gt'] += video_stats['gt_count']
            overall_stats['total_tp'] += video_stats['tp_count']
            overall_stats['total_fp'] += video_stats['fp_count']
            overall_stats['total_fn'] += video_stats['fn_count']

            overall_stats['per_video'][video_name] = {
                **video_stats,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            print(f"  TP: {video_stats['tp_count']}, FP: {video_stats['fp_count']}, FN: {video_stats['fn_count']}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        overall_precision = overall_stats['total_tp'] / (overall_stats['total_tp'] + overall_stats['total_fp']) if (
                                                                                                                           overall_stats[
                                                                                                                               'total_tp'] +
                                                                                                                           overall_stats[
                                                                                                                               'total_fp']) > 0 else 0
        overall_recall = overall_stats['total_tp'] / (overall_stats['total_tp'] + overall_stats['total_fn']) if (
                                                                                                                        overall_stats[
                                                                                                                            'total_tp'] +
                                                                                                                        overall_stats[
                                                                                                                            'total_fn']) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                              overall_precision + overall_recall) > 0 else 0

        summary_file = os.path.join(output_root, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'overall': {
                    'total_frames': overall_stats['total_frames'],
                    'total_gt': overall_stats['total_gt'],
                    'total_tp': overall_stats['total_tp'],
                    'total_fp': overall_stats['total_fp'],
                    'total_fn': overall_stats['total_fn'],
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1': overall_f1
                },
                'per_video': overall_stats['per_video']
            }, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"总体统计:")
        print(f"  总帧数: {overall_stats['total_frames']}")
        print(f"  总 GT 数: {overall_stats['total_gt']}")
        print(f"  TP: {overall_stats['total_tp']}")
        print(f"  FP: {overall_stats['total_fp']}")
        print(f"  FN: {overall_stats['total_fn']}")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall: {overall_recall:.4f}")
        print(f"  F1 Score: {overall_f1:.4f}")
        print(f"{'=' * 80}")
        print(f"结果已保存到: {output_root}")


def main():
    MODEL_PATH = r"C:/Users/win10/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"
    IMAGES_ROOT = r"E:/user/szx/sam/UA-DETRAC/DETRAC-Images"
    GT_XML_ROOT = r"E:/user/szx/sam/UA-DETRAC/DETRAC_XML"
    OUTPUT_ROOT = r"E:/user/szx/sam/UA-DETRAC/SAM3_Detection_Results_example_based2"

    tester = DETRACTester(
        model_path=MODEL_PATH,
        device='cuda',
        confidence_threshold=0.5,
        iou_threshold=0.5
    )

    tester.test_dataset(IMAGES_ROOT, GT_XML_ROOT, OUTPUT_ROOT)


if __name__ == '__main__':
    main()
