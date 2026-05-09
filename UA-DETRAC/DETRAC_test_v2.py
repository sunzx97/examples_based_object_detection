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
from sam3.model.sam3_image_processor import Sam3Processor

from sam3.model_builder import build_sam3_image_model


class DETRACTester:
    def __init__(self, model_path, device='cuda', confidence_threshold=0.5, iou_threshold=0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        print("加载 SAM3 模型...")
        self.model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=model_path,
            load_from_HF=False,
        )
        self.model.eval()
        print("模型加载完成!")

        self.text_prompt = "car or vehicle"

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

    def detect_frame(self, image_path, ignored_regions=None):
        """检测单帧图像"""
        image = Image.open(image_path).convert('RGB')

        processor = Sam3Processor(
            self.model,
            confidence_threshold=self.confidence_threshold
        )
        inference_state = processor.set_image(image)

        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state,
            prompt=self.text_prompt
        )

        boxes = inference_state.get('boxes', [])
        scores = inference_state.get('scores', [])

        detections = []
        if len(scores) > 0:
            for box, score in zip(boxes, scores):
                if score >= self.confidence_threshold:
                    box_xyxy = box.tolist()
                    x1, y1, x2, y2 = box_xyxy
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

                    detections.append({
                        'box': [left, top, width, height],
                        'confidence': float(score),
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

                try:
                    predictions = self.detect_frame(img_path, ignored_regions)

                    gt_targets = gt_data.get(frame_num, [])

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
    OUTPUT_ROOT = r"E:/user/szx/sam/UA-DETRAC/SAM3_Detection_Results"

    tester = DETRACTester(
        model_path=MODEL_PATH,
        device='cuda',
        confidence_threshold=0.5,
        iou_threshold=0.5
    )

    tester.test_dataset(IMAGES_ROOT, GT_XML_ROOT, OUTPUT_ROOT)


if __name__ == '__main__':
    main()
