import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

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

        # 初始化 INSID3 模型用于候选区域检测
        self._init_insid3()

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

    def _init_insid3(self):
        """初始化 INSID3 模型用于候选区域检测"""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from models import build_insid3
            from utils.data import build_transform, load_image

            print("初始化 INSID3 模型...")
            self.insid3_model = build_insid3(model_size='base', svd_components=20, device=self.device)
            self.insid3_model.eval()
            self.insid3_transform = build_transform(self.insid3_model.image_size)
            self.insid3_load_image = load_image
            self.insid3_available = True
            print("INSID3 模型初始化完成")
        except Exception as e:
            print(f"警告: INSID3 模型初始化失败 - {e}，将不使用候选区域检测")
            self.insid3_available = False

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

    def extract_candidate_regions_with_insid3(self, ref_image_path, target_image_path, confidence_threshold=0.7):
        """
        使用 INSID3 模型从参考图像到目标图像找出候选区域

        Args:
            ref_image_path: 参考图像路径（包含目标的示例）
            target_image_path: 目标图像路径（待检测的帧）
            confidence_threshold: 置信度阈值 (0-1)

        Returns:
            list: 候选边界框列表 [[x_min, y_min, x_max, y_max], ...]
        """
        if not self.insid3_available:
            return []

        try:
            from sklearn.cluster import DBSCAN

            # 加载图像
            ref_img, ref_orig_size = self.insid3_load_image(ref_image_path, self.insid3_transform, self.device)
            tgt_img, tgt_orig_size = self.insid3_load_image(target_image_path, self.insid3_transform, self.device)

            # 提取特征
            imgs = torch.cat([ref_img, tgt_img], dim=0).unsqueeze(0)
            fmaps_norm = F.normalize(self.insid3_model._extract_features(imgs), p=2, dim=2)

            # 使用去偏特征
            fmaps_norm = self.insid3_model._debias_features(fmaps_norm)

            feat_ref = fmaps_norm[0, 0]
            feat_tgt = fmaps_norm[0, 1]
            C, h, w = feat_ref.shape

            # 使用参考图像中心点作为查询点
            ref_center_x = ref_orig_size[1] // 2
            ref_center_y = ref_orig_size[0] // 2

            # 将中心点坐标转换到特征图尺度
            scale_h = h / ref_orig_size[0]
            scale_w = w / ref_orig_size[1]
            src_x = int(torch.round(torch.tensor(ref_center_x) * scale_w).item())
            src_y = int(torch.round(torch.tensor(ref_center_y) * scale_h).item())
            src_x = max(0, min(src_x, w - 1))
            src_y = max(0, min(src_y, h - 1))

            # 获取参考点特征
            src_desc = feat_ref[:, src_y, src_x]

            # 计算与 target 所有位置的相似度
            sim_map = torch.einsum('c,cxy->xy', src_desc, feat_tgt)

            # 归一化相似度到 [0, 1] 作为置信度
            sim_min = sim_map.min()
            sim_max = sim_map.max()
            if sim_max - sim_min > 1e-6:
                confidence_map = (sim_map - sim_min) / (sim_max - sim_min)
            else:
                confidence_map = torch.zeros_like(sim_map)

            # 找到所有超过阈值的点
            mask = confidence_map >= confidence_threshold
            if mask.sum() == 0:
                return []

            # 获取符合条件的点的坐标和置信度
            indices = torch.where(mask)
            pred_y = indices[0].float()
            pred_x = indices[1].float()
            confidences = confidence_map[mask]

            # 按置信度降序排序
            sorted_idx = torch.argsort(confidences, descending=True)
            pred_x = pred_x[sorted_idx]
            pred_y = pred_y[sorted_idx]
            confidences = confidences[sorted_idx]

            # 将坐标转换回原始图像尺度
            scale_h_orig = tgt_orig_size[0] / h
            scale_w_orig = tgt_orig_size[1] / w
            pred_x_orig = pred_x * scale_w_orig
            pred_y_orig = pred_y * scale_h_orig

            points = torch.stack([pred_x_orig, pred_y_orig], dim=1).cpu().numpy()

            # DBSCAN 聚类
            clustering = DBSCAN(eps=30.0, min_samples=3).fit(points)
            labels = clustering.labels_

            # 提取每个簇的边界框并扩展
            unique_labels = set(labels)
            candidate_boxes = []
            expand_w = ref_orig_size[1] // 2
            expand_h = ref_orig_size[0] // 2

            for label in unique_labels:
                if label == -1:
                    continue

                cluster_points = points[labels == label]
                x_min = float(np.min(cluster_points[:, 0]))
                y_min = float(np.min(cluster_points[:, 1]))
                x_max = float(np.max(cluster_points[:, 0]))
                y_max = float(np.max(cluster_points[:, 1]))

                # 向外扩展
                x_min -= expand_w
                y_min -= expand_h
                x_max += expand_w
                y_max += expand_h

                # 限制在图像边界内
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(tgt_orig_size[1], x_max)
                y_max = min(tgt_orig_size[0], y_max)

                candidate_boxes.append([x_min, y_min, x_max, y_max])

            return candidate_boxes

        except Exception as e:
            import traceback
            print(f"  INSID3 候选区域检测失败: {e}")
            print(f"  错误详情: {traceback.format_exc()}")
            return []

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

    def match_candidate_region_with_lightglue(self, ref_image_path, current_frame_path, candidate_box):
        """
        对候选区域使用 LightGlue 进行特征匹配，得到精确的检测框
        
        流程：
        1. 从当前帧中裁剪出候选区域
        2. 将裁剪区域与参考图像进行 LightGlue 匹配
        3. 通过单应性变换将参考图像中的目标映射到裁剪区域坐标
        4. 将裁剪区域坐标转换回原始图像坐标
        
        Args:
            ref_image_path: 参考图像路径（query 图像）
            current_frame_path: 当前帧路径
            candidate_box: 候选区域 [x_min, y_min, x_max, y_max]（在当前帧中的位置）
            
        Returns:
            list: [x_min, y_min, x_max, y_max] 或 None（原始图像坐标系）
        """
        if not self.lightglue_available:
            return None

        import cv2
        import tempfile

        try:
            # 读取当前帧和候选区域
            current_img_pil = Image.open(current_frame_path)
            curr_width, curr_height = current_img_pil.size
            
            x_min, y_min, x_max, y_max = candidate_box
            
            # 确保候选区域在图像范围内
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(curr_width, int(x_max))
            y_max = min(curr_height, int(y_max))
            
            if x_min >= x_max or y_min >= y_max:
                print(f"    候选区域无效: [{x_min}, {y_min}, {x_max}, {y_max}]")
                return None
            
            # 裁剪候选区域
            cropped_img = current_img_pil.crop((x_min, y_min, x_max, y_max))
            cropped_width, cropped_height = cropped_img.size
            
            if cropped_width == 0 or cropped_height == 0:
                print(f"    裁剪区域大小为0")
                return None
            
            # 将裁剪后的图像保存为临时文件（因为 LightGlue 的 load_image 需要文件路径）
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cropped_img.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name
            
            try:
                # 加载参考图像和裁剪后的候选区域
                ref_image = self.load_image(ref_image_path).cuda()
                cropped_tensor = self.load_image(tmp_path).cuda()
                
                # 提取特征
                ref_feats = self.extractor.extract(ref_image, resize=512)
                cropped_feats = self.extractor.extract(cropped_tensor, resize=512)

                # 进行特征匹配：ref -> cropped
                matches01 = self.matcher({'image0': ref_feats, 'image1': cropped_feats})
                ref_feats_rbd, cropped_feats_rbd, matches01_rbd = [self.rbd(x) for x in [ref_feats, cropped_feats, matches01]]
                matches = matches01_rbd['matches']

                if len(matches) < 50:
                    print(f"    LightGlue 匹配点不足: {len(matches)} < 4")
                    return None

                # 获取匹配的关键点
                kpts0 = ref_feats_rbd['keypoints']  # 参考图像关键点
                kpts1 = cropped_feats_rbd['keypoints']  # 裁剪区域关键点
                m_kpts0 = kpts0[matches[..., 0]]
                m_kpts1 = kpts1[matches[..., 1]]

                pts0 = m_kpts0.cpu().numpy()  # 参考图像中的点
                pts1 = m_kpts1.cpu().numpy()  # 裁剪区域中的点

                # 计算从参考图像到裁剪区域的单应性矩阵
                H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)

                if H is None:
                    print("    单应性矩阵计算失败")
                    return None

                # 读取参考图像尺寸
                ref_img_pil = Image.open(ref_image_path)
                ref_width, ref_height = ref_img_pil.size

                # 将参考图像的四个角点映射到裁剪区域
                corners_ref = np.float32([
                    [0, 0],
                    [ref_width, 0],
                    [ref_width, ref_height],
                    [0, ref_height]
                ]).reshape(-1, 1, 2)

                # 使用单应性矩阵将参考图像角点映射到裁剪区域坐标
                corners_cropped = cv2.perspectiveTransform(corners_ref, H).reshape(4, 2)

                # 计算在裁剪区域中的边界框
                crop_x_coords = corners_cropped[:, 0]
                crop_y_coords = corners_cropped[:, 1]
                crop_x_min = float(crop_x_coords.min())
                crop_y_min = float(crop_y_coords.min())
                crop_x_max = float(crop_x_coords.max())
                crop_y_max = float(crop_y_coords.max())

                # 将裁剪区域坐标转换回原始图像坐标
                orig_x_min = x_min + crop_x_min
                orig_y_min = y_min + crop_y_min
                orig_x_max = x_min + crop_x_max
                orig_y_max = y_min + crop_y_max

                # 限制在原始图像边界内
                orig_x_min = max(0, int(orig_x_min))
                orig_y_min = max(0, int(orig_y_min))
                orig_x_max = min(curr_width, int(orig_x_max))
                orig_y_max = min(curr_height, int(orig_y_max))

                # 验证边界框有效性
                if orig_x_min >= orig_x_max or orig_y_min >= orig_y_max:
                    print(f"    无效的边界框: [{orig_x_min}, {orig_y_min}, {orig_x_max}, {orig_y_max}]")
                    return None

                result_box = [orig_x_min, orig_y_min, orig_x_max, orig_y_max]
                print(f"    LightGlue 映射结果（原始坐标）: {result_box}")
                return result_box
            finally:
                # 删除临时文件
                import os
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        except Exception as e:
            import traceback
            print(f"  候选区域 LightGlue 匹配失败: {e}")
            print(f"  错误详情: {traceback.format_exc()}")
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

    def visualize_detections(self, image_path, detections, output_path, lightglue_matches=None, candidate_regions=None):
        """
        将检测框绘制到图像上并保存

        Args:
            image_path: 原始图像路径
            detections: 检测结果列表，每个元素包含 'box' 和 'confidence'
            output_path: 输出图像保存路径
            lightglue_matches: LightGlue 匹配结果列表 [x_min, y_min, x_max, y_max]
            candidate_regions: INSID3 候选区域列表 [x_min, y_min, x_max, y_max]
        """
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # 尝试加载字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            except:
                font = ImageFont.load_default()

            # 绘制 INSID3 候选区域（黄色虚线框）
            if candidate_regions:
                for idx, box in enumerate(candidate_regions):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
                    draw.text((x1, y1 - 35), f"INSID3-{idx+1}", fill="yellow", font=font)

            # 绘制 LightGlue 匹配结果（蓝色框）
            if lightglue_matches:
                for idx, box in enumerate(lightglue_matches):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
                    draw.text((x1, y1 - 20), f"LG-{idx+1}", fill="blue", font=font)

            # 绘制最终检测结果（红色框）
            colors = ["red", "green", "orange", "purple", "cyan", "magenta"]
            for idx, det in enumerate(detections):
                box = det['box']  # [left, top, width, height]
                conf = det['confidence']

                left, top, width, height = box
                x1, y1 = left, top
                x2, y2 = left + width, top + height

                color = colors[idx % len(colors)]

                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # 绘制置信度标签
                label = f"{conf:.2f}"
                text_bbox = draw.textbbox((x1, y1 - 50), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1 - 50), label, fill="white", font=font)

            # 保存图像
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)

        except Exception as e:
            import traceback
            print(f"  可视化保存失败: {e}")
            print(f"  错误详情: {traceback.format_exc()}")


    def detect_frame(self, image_path, ignored_regions=None, lightglue_configs=None, ref_image_path=None, frame_num=None, video_name=None, vis_output_dir=None):
        """检测单帧图像

        Args:
            image_path: 图像路径
            ignored_regions: 忽略区域列表
            lightglue_configs: LightGlue 配置列表（用于逐帧匹配）
            ref_image_path: 参考图像路径（用于 INSID3 候选区域检测）
            frame_num: 帧号（用于可视化保存）
            video_name: 视频名称（用于可视化保存）
            vis_output_dir: 可视化输出目录
        """
        # 第一步：使用 INSID3 找出候选区域
        candidate_regions = []
        if ref_image_path and self.insid3_available:
            try:
                candidate_regions = self.extract_candidate_regions_with_insid3(
                    ref_image_path, image_path, confidence_threshold=0.7
                )
                if candidate_regions:
                    print(f"  INSID3 找到 {len(candidate_regions)} 个候选区域")
            except Exception as e:
                import traceback
                print(f"  INSID3 候选区域检测失败: {e}")
                print(f"  错误详情: {traceback.format_exc()}")

        print("%"*20)
        print("candidate_regions:", candidate_regions)
        
        # 第二步：对每个候选区域使用 LightGlue 进行特征匹配，得到精确的 bbox
        final_bboxes = []
        if candidate_regions and ref_image_path and self.lightglue_available:
            try:
                for idx, candidate_box in enumerate(candidate_regions):
                    matched_bbox = self.match_candidate_region_with_lightglue(
                        ref_image_path, image_path, candidate_box
                    )
                    if matched_bbox is not None:
                        final_bboxes.append(matched_bbox)
                        print(f"  候选区域 {idx+1} LightGlue 匹配成功: {matched_bbox}")
            except Exception as e:
                import traceback
                print(f"  候选区域 LightGlue 匹配失败: {e}")
                print(f"  错误详情: {traceback.format_exc()}")

        print("final_bboxes after LightGlue:", final_bboxes)

        # 第三步：进行 SAM3 检测
        # 设置图像
        self.predictor.set_image(image_path)

        # 准备 bbox 提示
        bboxes_array = None
        labels_array = None
        if final_bboxes:
            bboxes_array = np.array(final_bboxes, dtype=np.float32)
            labels_array = np.ones(len(final_bboxes), dtype=np.int32)

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

        # 可视化并保存检测结果
        if vis_output_dir and frame_num is not None and video_name is not None:
            vis_filename = f"frame_{frame_num:05d}_det.jpg"
            vis_path = os.path.join(vis_output_dir, video_name, vis_filename)
            self.visualize_detections(
                image_path, 
                detections, 
                vis_path,
                lightglue_matches=final_bboxes,
                candidate_regions=candidate_regions
            )

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

    def test_dataset(self, images_root, gt_xml_root, output_root, vis_output_dir=None):
        """
        测试整个 DETRAC 数据集
        """
        os.makedirs(output_root, exist_ok=True)
        
        # 创建可视化输出目录
        if vis_output_dir:
            os.makedirs(vis_output_dir, exist_ok=True)
            print(f"可视化结果将保存到: {vis_output_dir}")

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

            # 确定参考图像路径（用于 INSID3）
            ref_image_path = None
            if self.insid3_available and lightglue_configs:
                # 使用第一个 LightGlue 配置的图像作为参考
                ref_image_path = lightglue_configs[0].get('imagePath', None)
                if ref_image_path:
                    print(f"  使用参考图像: {ref_image_path}")

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
                    predictions = self.detect_frame(
                        img_path,
                        ignored_regions,
                        lightglue_configs,
                        ref_image_path,
                        frame_num=frame_num,
                        video_name=video_name,
                        vis_output_dir=vis_output_dir
                    )

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
    OUTPUT_ROOT = r"E:/user/szx/sam/UA-DETRAC/SAM3_Detection_Results_example_based3"
    VIS_OUTPUT_DIR = r"E:/user/szx/sam/UA-DETRAC/SAM3_Visualization_Results"

    tester = DETRACTester(
        model_path=MODEL_PATH,
        device='cuda',
        confidence_threshold=0.5,
        iou_threshold=0.5
    )

    tester.test_dataset(IMAGES_ROOT, GT_XML_ROOT, OUTPUT_ROOT, vis_output_dir=VIS_OUTPUT_DIR)


if __name__ == '__main__':
    main()
