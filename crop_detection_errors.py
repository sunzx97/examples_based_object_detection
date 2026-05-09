#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从SAM3检测结果XML文件中裁剪漏检(FN)和误检(FP)的图像区域
并生成JSON记录文件
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def parse_xml(xml_path):
    """
    解析XML文件，提取每一帧的FN和FP信息
    
    Returns:
        dict: {
            frame_num: {
                'FN': [{'target_id': id, 'box': [left, top, width, height]}],
                'FP': [{'box': [left, top, width, height], 'confidence': conf}]
            }
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    frame_data = {}
    
    for frame_elem in root.findall('frame'):
        frame_num = int(frame_elem.get('num'))
        target_list = frame_elem.find('target_list')
        
        if target_list is None:
            continue
        
        fn_boxes = []
        fp_boxes = []
        
        for target_elem in target_list.findall('target'):
            box_elem = target_elem.find('box')
            if box_elem is None:
                continue
            
            left = float(box_elem.get('left'))
            top = float(box_elem.get('top'))
            width = float(box_elem.get('width'))
            height = float(box_elem.get('height'))
            
            attr_elem = target_elem.find('attribute')
            if attr_elem is None:
                continue
            
            detection_status = attr_elem.get('detection_status', '')
            
            if detection_status == 'FN':
                # 漏检：有target id
                target_id = target_elem.get('id')
                fn_boxes.append({
                    'target_id': target_id,
                    'box': [left, top, width, height]
                })
            elif detection_status == 'FP':
                # 误检：没有target id
                confidence = float(attr_elem.get('confidence', 0))
                fp_boxes.append({
                    'box': [left, top, width, height],
                    'confidence': confidence
                })
        
        if fn_boxes or fp_boxes:
            frame_data[frame_num] = {
                'FN': fn_boxes,
                'FP': fp_boxes
            }
    
    return frame_data


def crop_and_save(image_path, box, save_path):
    """
    根据边界框裁剪图像并保存
    
    Args:
        image_path: 原始图像路径
        box: [left, top, width, height]
        save_path: 保存路径
    """
    try:
        img = Image.open(image_path)
        left, top, width, height = box
        
        # 转换为(x1, y1, x2, y2)格式
        x1 = max(0, int(left))
        y1 = max(0, int(top))
        x2 = min(img.width, int(left + width))
        y2 = min(img.height, int(top + height))
        
        # 确保裁剪区域有效
        if x2 <= x1 or y2 <= y1:
            print(f"警告: 无效的裁剪区域 {box} in {image_path}")
            return False
        
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(save_path)
        return True
    except Exception as e:
        print(f"裁剪图像失败 {image_path}: {e}")
        return False


def get_image_path(video_name, frame_num, video_frames_root):
    """
    获取视频帧图像的路径
    
    Args:
        video_name: 视频名称 (如 MVI_20011)
        frame_num: 帧号
        video_frames_root: 视频帧根目录
    
    Returns:
        str: 图像文件路径
    """
    # UA-DETRAC数据集的典型路径结构
    # 假设视频帧存储在 video_frames_root/video_name/ 目录下
    # 帧图像命名为 imgXXXX.jpg 或类似格式
    
    # 尝试几种常见的命名格式
    possible_paths = [
        os.path.join(video_frames_root, video_name, f"img{frame_num:05d}.jpg"),
        os.path.join(video_frames_root, video_name, f"img{frame_num:04d}.jpg"),
        os.path.join(video_frames_root, video_name, f"{frame_num:05d}.jpg"),
        os.path.join(video_frames_root, video_name, f"{frame_num:06d}.jpg"),
        os.path.join(video_frames_root, video_name, f"frame_{frame_num:05d}.jpg"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 如果都不存在，列出目录内容帮助调试
    video_dir = os.path.join(video_frames_root, video_name)
    if os.path.exists(video_dir):
        files = os.listdir(video_dir)[:5]  # 只列出前5个文件
        print(f"警告: 未找到帧 {frame_num} 的图像，目录 {video_dir} 中的文件示例: {files}")
    else:
        print(f"警告: 视频目录不存在: {video_dir}")
    
    return None


def process_video_xml(xml_path, video_frames_root, output_root):
    """
    处理单个视频的XML文件
    
    Args:
        xml_path: XML文件路径
        video_frames_root: 视频帧图像根目录
        output_root: 输出根目录
    
    Returns:
        dict: JSON记录数据
    """
    video_name = Path(xml_path).stem  # 去掉.xml后缀
    print(f"\n处理视频: {video_name}")
    
    # 创建输出目录
    video_output_dir = os.path.join(output_root, video_name)
    fn_dir = os.path.join(video_output_dir, "FN")
    fp_dir = os.path.join(video_output_dir, "FP")
    
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)
    
    # 解析XML
    frame_data = parse_xml(xml_path)
    
    # JSON记录
    json_record = {
        'video_name': video_name,
        'frames_with_errors': []
    }
    
    total_fn = 0
    total_fp = 0
    
    # 处理每一帧
    for frame_num, data in tqdm(frame_data.items(), desc=f"处理帧"):
        frame_record = {
            'frame_num': frame_num,
            'FN': [],
            'FP': []
        }
        
        # 获取图像路径
        image_path = get_image_path(video_name, frame_num, video_frames_root)
        print("image_path:", image_path)
        
        if image_path is None:
            print(f"跳过帧 {frame_num}: 找不到图像文件")
            continue
        
        # 处理FN（漏检）
        for idx, fn_info in enumerate(data['FN']):
            target_id = fn_info['target_id']
            box = fn_info['box']
            
            # 保存文件名: frame_XXX_target_YYY.jpg
            save_filename = f"frame_{frame_num:05d}_target_{target_id}.jpg"
            save_path = os.path.join(fn_dir, save_filename)
            
            if crop_and_save(image_path, box, save_path):
                total_fn += 1
                frame_record['FN'].append({
                    'target_id': target_id,
                    'box': box,
                    'image_path': os.path.relpath(save_path, output_root)
                })
        
        # 处理FP（误检）
        for idx, fp_info in enumerate(data['FP']):
            box = fp_info['box']
            confidence = fp_info['confidence']
            
            # 保存文件名: frame_XXX_fp_YYY_conf_ZZZZ.jpg
            save_filename = f"frame_{frame_num:05d}_fp_{idx:03d}_conf_{confidence:.4f}.jpg"
            save_path = os.path.join(fp_dir, save_filename)
            
            if crop_and_save(image_path, box, save_path):
                total_fp += 1
                frame_record['FP'].append({
                    'box': box,
                    'confidence': confidence,
                    'image_path': os.path.relpath(save_path, output_root)
                })
        
        # 只有当该帧有FN或FP时才添加到记录
        if frame_record['FN'] or frame_record['FP']:
            json_record['frames_with_errors'].append(frame_record)
    
    # 保存JSON记录
    json_path = os.path.join(video_output_dir, f"{video_name}_errors.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_record, f, indent=2, ensure_ascii=False)
    
    print(f"视频 {video_name} 处理完成:")
    print(f"  - FN (漏检) 数量: {total_fn}")
    print(f"  - FP (误检) 数量: {total_fp}")
    print(f"  - 输出目录: {video_output_dir}")
    print(f"  - JSON记录: {json_path}")
    
    return json_record


def main():
    """主函数"""
    # 配置路径
    xml_results_dir = r"E:/user/szx/sam/UA-DETRAC/SAM3_Detection_Results_example_based2"
    video_frames_root = r"E:/user/szx/sam/UA-DETRAC/DETRAC-Images"  # 需要根据实际情况修改
    output_root = "save_video_crop_img_example_based2"
    
    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = sorted([
        os.path.join(xml_results_dir, f) 
        for f in os.listdir(xml_results_dir) 
        if f.endswith('.xml')
    ])
    
    print(f"找到 {len(xml_files)} 个XML文件")
    print(f"视频帧根目录: {video_frames_root}")
    print(f"输出目录: {output_root}")
    
    # 处理所有XML文件
    all_records = []
    for xml_path in tqdm(xml_files, desc="处理视频"):
        try:
            record = process_video_xml(xml_path, video_frames_root, output_root)
            all_records.append(record)
        except Exception as e:
            print(f"处理 {xml_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总统计
    summary = {
        'total_videos': len(all_records),
        'videos': [
            {
                'video_name': r['video_name'],
                'frames_with_errors_count': len(r['frames_with_errors'])
            }
            for r in all_records
        ]
    }
    
    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n所有视频处理完成!")
    print(f"汇总统计保存在: {summary_path}")


if __name__ == "__main__":
    main()
