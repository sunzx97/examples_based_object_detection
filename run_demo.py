import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import cv2
import gc


class SingleImageDetector:
    def __init__(self, sam3_model_path, device='cuda'):
        self.device = device

        # 初始化组件（按需加载）
        self.sam3_predictor = None
        self.insid3_model = None
        self.lightglue_extractor = None
        self.lightglue_matcher = None

        self.sam3_model_path = sam3_model_path

    def init_sam3(self):
        """初始化 SAM3 模型"""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            print("初始化 SAM3...")
            overrides = dict(
                conf=0.5,
                task="segment",
                mode="predict",
                model=self.sam3_model_path,
                half=False,
                save=False,
            )
            self.sam3_predictor = SAM3SemanticPredictor(overrides=overrides)
            print("SAM3 初始化完成")
        except Exception as e:
            print(f"SAM3 初始化失败: {e}")
            raise

    def init_insid3(self):
        """初始化 INSID3 模型"""
        try:
            from models import build_insid3
            from utils.data import build_transform, load_image

            print("初始化 INSID3...")
            self.insid3_model = build_insid3(model_size='base', svd_components=20, device=self.device)
            self.insid3_model.eval()
            self.insid3_transform = build_transform(self.insid3_model.image_size)
            self.insid3_load_image = load_image
            print("INSID3 初始化完成")
        except Exception as e:
            print(f"INSID3 初始化失败: {e}")
            raise

    def init_lightglue(self):
        """初始化 LightGlue"""
        try:
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import load_image, rbd

            print("初始化 LightGlue...")
            self.lightglue_extractor = SuperPoint(max_num_keypoints=4096).eval().cuda()
            self.lightglue_matcher = LightGlue(
                features='superpoint',
                depth_confidence=0.95,
                width_confidence=0.95
            ).eval().cuda()
            self.lightglue_load_image = load_image
            self.lightglue_rbd = rbd
            print("LightGlue 初始化完成")
        except Exception as e:
            print(f"LightGlue 初始化失败: {e}")
            raise

    def release_sam3(self):
        """释放 SAM3 显存"""
        if self.sam3_predictor is not None:
            del self.sam3_predictor
            self.sam3_predictor = None
            torch.cuda.empty_cache()
            gc.collect()
            print("SAM3 显存已释放")

    def release_insid3(self):
        """释放 INSID3 显存"""
        if self.insid3_model is not None:
            del self.insid3_model
            self.insid3_model = None
            torch.cuda.empty_cache()
            gc.collect()
            print("INSID3 显存已释放")

    def release_lightglue(self):
        """释放 LightGlue 显存"""
        if self.lightglue_extractor is not None:
            del self.lightglue_extractor
            del self.lightglue_matcher
            self.lightglue_extractor = None
            self.lightglue_matcher = None
            torch.cuda.empty_cache()
            gc.collect()
            print("LightGlue 显存已释放")

    def extract_candidate_regions_with_insid3(self, ref_image_path, target_image_path, confidence_threshold=0.7):
        """
        使用 INSID3 提取候选区域

        Args:
            ref_image_path: 参考图像路径（query）
            target_image_path: 目标图像路径（target）
            confidence_threshold: 置信度阈值

        Returns:
            list: 候选框列表 [[x_min, y_min, x_max, y_max], ...]
        """
        try:
            from sklearn.cluster import DBSCAN

            # 加载图像
            ref_img, ref_orig_size = self.insid3_load_image(ref_image_path, self.insid3_transform, self.device)
            tgt_img, tgt_orig_size = self.insid3_load_image(target_image_path, self.insid3_transform, self.device)

            # 提取特征
            imgs = torch.cat([ref_img, tgt_img], dim=0).unsqueeze(0)
            fmaps_norm = F.normalize(self.insid3_model._extract_features(imgs), p=2, dim=2)
            fmaps_norm = self.insid3_model._debias_features(fmaps_norm)

            feat_ref = fmaps_norm[0, 0]
            feat_tgt = fmaps_norm[0, 1]
            C, h, w = feat_ref.shape

            # 使用参考图像中心点作为查询点
            ref_center_x = ref_orig_size[1] // 2
            ref_center_y = ref_orig_size[0] // 2

            scale_h = h / ref_orig_size[0]
            scale_w = w / ref_orig_size[1]
            src_x = int(torch.round(torch.tensor(ref_center_x) * scale_w).item())
            src_y = int(torch.round(torch.tensor(ref_center_y) * scale_h).item())
            src_x = max(0, min(src_x, w - 1))
            src_y = max(0, min(src_y, h - 1))

            # 获取参考点特征并计算相似度
            src_desc = feat_ref[:, src_y, src_x]
            sim_map = torch.einsum('c,cxy->xy', src_desc, feat_tgt)

            # 归一化相似度
            sim_min = sim_map.min()
            sim_max = sim_map.max()
            if sim_max - sim_min > 1e-6:
                confidence_map = (sim_map - sim_min) / (sim_max - sim_min)
            else:
                confidence_map = torch.zeros_like(sim_map)

            # 找到超过阈值的点
            mask = confidence_map >= confidence_threshold
            if mask.sum() == 0:
                return []

            indices = torch.where(mask)
            pred_y = indices[0].float()
            pred_x = indices[1].float()
            confidences = confidence_map[mask]

            # 按置信度排序
            sorted_idx = torch.argsort(confidences, descending=True)
            pred_x = pred_x[sorted_idx]
            pred_y = pred_y[sorted_idx]

            # 转换回原始尺度
            scale_h_orig = tgt_orig_size[0] / h
            scale_w_orig = tgt_orig_size[1] / w
            pred_x_orig = pred_x * scale_w_orig
            pred_y_orig = pred_y * scale_h_orig

            points = torch.stack([pred_x_orig, pred_y_orig], dim=1).cpu().numpy()

            # DBSCAN 聚类
            clustering = DBSCAN(eps=30.0, min_samples=3).fit(points)
            labels = clustering.labels_

            # 提取边界框并扩展
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
            print(f"INSID3 候选区域检测失败: {e}")
            print(traceback.format_exc())
            return []

    def match_candidate_region_with_lightglue(self, ref_image_path, target_image_path, candidate_box):
        """
        对候选区域使用 LightGlue 进行精确匹配

        Args:
            ref_image_path: 参考图像路径（query）
            target_image_path: 目标图像路径（target）
            candidate_box: 候选区域 [x_min, y_min, x_max, y_max]

        Returns:
            list: [x_min, y_min, x_max, y_max] 或 None
        """
        try:
            # 读取当前帧和候选区域
            target_img_pil = Image.open(target_image_path).convert("RGB")
            curr_width, curr_height = target_img_pil.size

            x_min, y_min, x_max, y_max = candidate_box

            # 确保候选区域在图像范围内
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(curr_width, int(x_max))
            y_max = min(curr_height, int(y_max))

            if x_min >= x_max or y_min >= y_max:
                print(f"候选区域无效: [{x_min}, {y_min}, {x_max}, {y_max}]")
                return None

            # 裁剪候选区域
            cropped_img = target_img_pil.crop((x_min, y_min, x_max, y_max))
            cropped_width, cropped_height = cropped_img.size

            if cropped_width == 0 or cropped_height == 0:
                print("裁剪区域大小为0")
                return None

            # 确保是 RGB 模式
            if cropped_img.mode != 'RGB':
                cropped_img = cropped_img.convert('RGB')

            # 保存为临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cropped_img.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name

            try:
                # 加载图像
                ref_image = self.lightglue_load_image(ref_image_path).cuda()
                cropped_tensor = self.lightglue_load_image(tmp_path).cuda()

                # 提取特征
                ref_feats = self.lightglue_extractor.extract(ref_image, resize=512)
                cropped_feats = self.lightglue_extractor.extract(cropped_tensor, resize=512)

                # 特征匹配
                matches01 = self.lightglue_matcher({'image0': ref_feats, 'image1': cropped_feats})
                ref_feats_rbd, cropped_feats_rbd, matches01_rbd = [
                    self.lightglue_rbd(x) for x in [ref_feats, cropped_feats, matches01]
                ]
                matches = matches01_rbd['matches']

                if len(matches) < 50:
                    print(f"LightGlue 匹配点不足: {len(matches)} < 50")
                    return None

                # 获取匹配的关键点
                kpts0 = ref_feats_rbd['keypoints']
                kpts1 = cropped_feats_rbd['keypoints']
                m_kpts0 = kpts0[matches[..., 0]]
                m_kpts1 = kpts1[matches[..., 1]]

                pts0 = m_kpts0.cpu().numpy()
                pts1 = m_kpts1.cpu().numpy()

                # 计算单应性矩阵
                H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)

                if H is None:
                    print("单应性矩阵计算失败")
                    return None

                # 读取参考图像尺寸
                ref_img_pil = Image.open(ref_image_path).convert("RGB")
                ref_width, ref_height = ref_img_pil.size

                # 将参考图像的角点映射到裁剪区域
                corners_ref = np.float32([
                    [0, 0],
                    [ref_width, 0],
                    [ref_width, ref_height],
                    [0, ref_height]
                ]).reshape(-1, 1, 2)

                corners_cropped = cv2.perspectiveTransform(corners_ref, H).reshape(4, 2)

                # 计算裁剪区域中的边界框
                crop_x_coords = corners_cropped[:, 0]
                crop_y_coords = corners_cropped[:, 1]
                crop_x_min = float(crop_x_coords.min())
                crop_y_min = float(crop_y_coords.min())
                crop_x_max = float(crop_x_coords.max())
                crop_y_max = float(crop_y_coords.max())

                # 转换回原始图像坐标
                orig_x_min = x_min + crop_x_min
                orig_y_min = y_min + crop_y_min
                orig_x_max = x_min + crop_x_max
                orig_y_max = y_min + crop_y_max

                # 限制在原始图像边界内
                orig_x_min = max(0, int(orig_x_min))
                orig_y_min = max(0, int(orig_y_min))
                orig_x_max = min(curr_width, int(orig_x_max))
                orig_y_max = min(curr_height, int(orig_y_max))

                if orig_x_min >= orig_x_max or orig_y_min >= orig_y_max:
                    print(f"无效的边界框: [{orig_x_min}, {orig_y_min}, {orig_x_max}, {orig_y_max}]")
                    return None

                result_box = [orig_x_min, orig_y_min, orig_x_max, orig_y_max]
                print(f"LightGlue 映射结果: {result_box}")
                return result_box

            finally:
                # 删除临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            import traceback
            print(f"LightGlue 匹配失败: {e}")
            print(traceback.format_exc())
            return None

    def detect_with_sam3(self, image_path, bboxes, text_prompt):
        """
        使用 SAM3 进行检测

        Args:
            image_path: 图像路径
            bboxes: bbox 列表 [[x_min, y_min, x_max, y_max], ...]
            text_prompt: 文本提示

        Returns:
            list: 检测结果 [{'box': [left, top, width, height], 'confidence': float}, ...]
        """
        try:
            # 设置图像
            self.sam3_predictor.set_image(image_path)

            # 准备 bbox 提示
            bboxes_array = None
            labels_array = None
            if bboxes:
                bboxes_array = np.array(bboxes, dtype=np.float32)
                labels_array = np.ones(len(bboxes), dtype=np.int32)

            # 执行检测
            results = self.sam3_predictor(
                text=[text_prompt],
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

                        left = float(x1)
                        top = float(y1)
                        width = float(x2 - x1)
                        height = float(y2 - y1)

                        # 过滤过小的检测框
                        if width * height < 100:
                            continue

                        detections.append({
                            'box': [left, top, width, height],
                            'confidence': float(conf),
                        })

            return detections

        except Exception as e:
            import traceback
            print(f"SAM3 检测失败: {e}")
            print(traceback.format_exc())
            return []

    def visualize_results(self, image_path, detections, output_path, candidate_regions=None, lightglue_bboxes=None):
        """
        可视化检测结果

        Args:
            image_path: 原始图像路径
            detections: 最终检测结果
            output_path: 输出路径
            candidate_regions: INSID3 候选区域
            lightglue_bboxes: LightGlue 匹配后的 bbox
        """
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # 尝试加载字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            except:
                font = ImageFont.load_default()

            # 绘制 INSID3 候选区域（黄色）
            if candidate_regions:
                for idx, box in enumerate(candidate_regions):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
                    draw.text((x1, y1 - 35), f"Candidate-{idx + 1}", fill="yellow", font=font)

            # 绘制 LightGlue 匹配结果（蓝色）
            if lightglue_bboxes:
                for idx, box in enumerate(lightglue_bboxes):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
                    draw.text((x1, y1 - 20), f"LightGlue-{idx + 1}", fill="blue", font=font)

            # 绘制最终检测结果（红色）
            colors = ["red", "green", "orange", "purple", "cyan", "magenta"]
            for idx, det in enumerate(detections):
                box = det['box']
                conf = det['confidence']

                left, top, width, height = box
                x1, y1 = left, top
                x2, y2 = left + width, top + height

                color = colors[idx % len(colors)]

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                label = f"{conf:.2f}"
                text_bbox = draw.textbbox((x1, y1 - 50), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1 - 50), label, fill="white", font=font)

            # 保存
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            img.save(output_path)
            print(f"可视化结果保存到: {output_path}")

        except Exception as e:
            import traceback
            print(f"可视化失败: {e}")
            print(traceback.format_exc())

    def run(self, query_image_path, target_image_path, text_prompt, output_path="output_result.jpg"):
        """
        完整的检测流程

        Args:
            query_image_path: query 图像路径（参考图像）
            target_image_path: target 图像路径（待检测图像）
            text_prompt: 文本提示（如 "car or vehicle"）
            output_path: 可视化结果输出路径

        Returns:
            list: 最终检测结果
        """
        print("=" * 80)
        print("开始单图像检测流程")
        print("=" * 80)

        # ========== 第一步：INSID3 候选区域检测 ==========
        print("\n【步骤 1】INSID3 候选区域检测...")
        self.init_insid3()

        candidate_regions = self.extract_candidate_regions_with_insid3(
            query_image_path,
            target_image_path,
            confidence_threshold=0.7
        )

        print(f"INSID3 找到 {len(candidate_regions)} 个候选区域:")
        for i, box in enumerate(candidate_regions):
            print(f"  候选区域 {i + 1}: {box}")

        # 释放 INSID3 显存
        self.release_insid3()

        # ========== 第二步：LightGlue 精确匹配 ==========
        print("\n【步骤 2】LightGlue 精确匹配...")
        final_bboxes = []

        if candidate_regions:
            self.init_lightglue()

            for idx, candidate_box in enumerate(candidate_regions):
                print(f"  处理候选区域 {idx + 1}/{len(candidate_regions)}...")
                matched_bbox = self.match_candidate_region_with_lightglue(
                    query_image_path,
                    target_image_path,
                    candidate_box
                )
                if matched_bbox is not None:
                    final_bboxes.append(matched_bbox)
                    print(f"    ✓ 匹配成功: {matched_bbox}")
                else:
                    print(f"    ✗ 匹配失败")

            # 释放 LightGlue 显存
            self.release_lightglue()

        print(f"\nLightGlue 匹配得到 {len(final_bboxes)} 个精确 bbox:")
        for i, box in enumerate(final_bboxes):
            print(f"  BBox {i + 1}: {box}")

        # ========== 第三步：SAM3 分割检测 ==========
        print("\n【步骤 3】SAM3 分割检测...")
        self.init_sam3()

        detections = self.detect_with_sam3(
            target_image_path,
            final_bboxes,
            text_prompt
        )

        print(f"\nSAM3 检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            print(f"  目标 {i + 1}: box={det['box']}, confidence={det['confidence']:.2f}")

        # 释放 SAM3 显存
        self.release_sam3()

        # ========== 第四步：可视化结果 ==========
        print("\n【步骤 4】生成可视化结果...")
        self.visualize_results(
            target_image_path,
            detections,
            output_path,
            candidate_regions=candidate_regions,
            lightglue_bboxes=final_bboxes
        )

        print("\n" + "=" * 80)
        print("检测流程完成！")
        print("=" * 80)

        return detections


def main():
    """主函数"""
    # 配置参数
    SAM3_MODEL_PATH = "sam3.pt"  # 修改为你的 SAM3 模型路径
    QUERY_IMAGE_PATH = "query.png"  # 修改为你的 query 图像路径
    TARGET_IMAGE_PATH = "target3.png"  # 修改为你的 target 图像路径
    TEXT_PROMPT = "vehicle"  # 修改为你的文本提示
    OUTPUT_PATH = "./detection_result.jpg"  # 输出可视化结果路径

    # 创建检测器
    detector = SingleImageDetector(
        sam3_model_path=SAM3_MODEL_PATH,
        device='cuda'
    )

    # 运行检测流程
    results = detector.run(
        query_image_path=QUERY_IMAGE_PATH,
        target_image_path=TARGET_IMAGE_PATH,
        text_prompt=TEXT_PROMPT,
        output_path=OUTPUT_PATH
    )

    print(f"\n最终检测结果数量: {len(results)}")


if __name__ == '__main__':
    main()
