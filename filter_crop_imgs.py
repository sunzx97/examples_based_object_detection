#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据JSON记录文件，筛选并重新裁剪宽度>100且高度>80的图像区域
"""

import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def filter_and_recrop(input_root, output_root, min_width=100, min_height=80):
    """
    读取JSON文件，筛选符合条件的检测框并重新裁剪

    Args:
        input_root: 原始裁剪图像的根目录（包含JSON文件）
        output_root: 输出目录
        min_width: 最小宽度阈值
        min_height: 最小高度阈值
    """
    # 查找所有JSON文件
    json_files = []
    for root_dir, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('_errors.json'):
                json_files.append(os.path.join(root_dir, file))

    print(f"找到 {len(json_files)} 个JSON文件")

    total_filtered = 0
    total_skipped = 0

    for json_path in tqdm(json_files, desc="处理视频"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            video_name = json_data['video_name']
            print(f"\n处理视频: {video_name}")

            # 创建输出目录
            video_output_dir = os.path.join(output_root, video_name)
            fn_dir = os.path.join(video_output_dir, "FN")
            fp_dir = os.path.join(video_output_dir, "FP")

            os.makedirs(fn_dir, exist_ok=True)
            os.makedirs(fp_dir, exist_ok=True)

            # 获取原始图像根目录（从JSON路径推断）
            # 假设JSON在 input_root/video_name/ 下
            video_input_dir = os.path.dirname(json_path)

            filtered_record = {
                'video_name': video_name,
                'min_width': min_width,
                'min_height': min_height,
                'frames_with_errors': []
            }

            # 遍历每一帧
            for frame_info in json_data.get('frames_with_errors', []):
                frame_num = frame_info['frame_num']

                # 获取原始图像路径（需要从box坐标反推）
                # 由于JSON中保存的是相对路径，我们需要找到对应的原始视频帧
                # 这里假设原始视频帧在 DETRAC-Images/video_name/ 目录下
                original_image_path = get_original_image_path(video_name, frame_num)

                if original_image_path is None or not os.path.exists(original_image_path):
                    print(f"  跳过帧 {frame_num}: 找不到原始图像")
                    continue

                frame_record = {
                    'frame_num': frame_num,
                    'FN': [],
                    'FP': []
                }

                # 处理FN
                for fn_info in frame_info.get('FN', []):
                    box = fn_info['box']
                    width = box[2]
                    height = box[3]

                    if width > min_width and height > min_height:
                        target_id = fn_info['target_id']

                        # 重新裁剪
                        save_filename = f"frame_{frame_num:05d}_target_{target_id}.jpg"
                        save_path = os.path.join(fn_dir, save_filename)

                        if crop_from_original(original_image_path, box, save_path):
                            filtered_record['frames_with_errors'].append(frame_record) if frame_record not in \
                                                                                          filtered_record[
                                                                                              'frames_with_errors'] else None
                            frame_record['FN'].append({
                                'target_id': target_id,
                                'box': box,
                                'width': width,
                                'height': height,
                                'image_path': os.path.relpath(save_path, output_root)
                            })
                            total_filtered += 1
                        else:
                            total_skipped += 1
                    else:
                        total_skipped += 1

                # 处理FP
                for fp_info in frame_info.get('FP', []):
                    box = fp_info['box']
                    width = box[2]
                    height = box[3]

                    if width > min_width and height > min_height:
                        confidence = fp_info['confidence']

                        # 重新裁剪
                        save_filename = f"frame_{frame_num:05d}_fp_conf_{confidence:.4f}.jpg"
                        save_path = os.path.join(fp_dir, save_filename)

                        if crop_from_original(original_image_path, box, save_path):
                            if frame_record not in filtered_record['frames_with_errors']:
                                filtered_record['frames_with_errors'].append(frame_record)
                            frame_record['FP'].append({
                                'box': box,
                                'confidence': confidence,
                                'width': width,
                                'height': height,
                                'image_path': os.path.relpath(save_path, output_root)
                            })
                            total_filtered += 1
                        else:
                            total_skipped += 1
                    else:
                        total_skipped += 1

            # 保存过滤后的JSON记录
            if filtered_record['frames_with_errors']:
                json_output_path = os.path.join(video_output_dir, f"{video_name}_filtered_errors.json")
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_record, f, indent=2, ensure_ascii=False)

                print(f"  视频 {video_name} 过滤完成:")
                print(f"    - 保留的检测框数量: {total_filtered}")
                print(f"    - 过滤掉的检测框数量: {total_skipped}")
                print(f"    - JSON记录: {json_output_path}")

        except Exception as e:
            print(f"处理 {json_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n所有视频处理完成!")
    print(f"总共保留: {total_filtered} 个检测框")
    print(f"总共过滤: {total_skipped} 个检测框")


def get_original_image_path(video_name, frame_num):
    """
    获取原始视频帧图像的路径

    Args:
        video_name: 视频名称
        frame_num: 帧号

    Returns:
        str: 图像路径或None
    """
    # UA-DETRAC数据集的典型路径
    possible_roots = [
        r"E:/user/szx/sam/UA-DETRAC/DETRAC-Images",
        "/home/sun/data/datasets/UA-DETRAC/Insight-MVT_Annotation_Train",
    ]

    for video_frames_root in possible_roots:
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

    return None


def crop_from_original(image_path, box, save_path):
    """
    从原始图像裁剪指定区域

    Args:
        image_path: 原始图像路径
        box: [left, top, width, height]
        save_path: 保存路径

    Returns:
        bool: 是否成功
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
            print(f"警告: 无效的裁剪区域 {box}")
            return False

        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(save_path)
        return True
    except Exception as e:
        print(f"裁剪图像失败 {image_path}: {e}")
        return False


def main():
    """主函数"""
    # 配置路径
    input_root = "save_video_crop_img_example_based2"  # 包含JSON文件的目录
    output_root = "save_video_crop_img_filtered_example_based2"  # 过滤后的输出目录

    # 尺寸阈值
    min_width = 100
    min_height = 80

    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"尺寸阈值: 宽度>{min_width}, 高度>{min_height}")

    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)

    # 执行过滤和重新裁剪
    filter_and_recrop(input_root, output_root, min_width, min_height)


if __name__ == "__main__":
    main()
