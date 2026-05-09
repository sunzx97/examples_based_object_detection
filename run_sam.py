import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SAM3OnlyDetector:
    def __init__(self, sam3_model_path, device='cuda', confidence_threshold=0.5):
        """
        初始化仅使用 SAM3 的检测器

        Args:
            sam3_model_path: SAM3 模型路径
            device: 设备 ('cuda' 或 'cpu')
            confidence_threshold: 置信度阈值
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.sam3_model_path = sam3_model_path
        self.sam3_predictor = None

    def init_sam3(self):
        """初始化 SAM3 模型"""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            print("初始化 SAM3...")
            overrides = dict(
                conf=self.confidence_threshold,
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

    def release_sam3(self):
        """释放 SAM3 显存"""
        if self.sam3_predictor is not None:
            del self.sam3_predictor
            self.sam3_predictor = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("SAM3 显存已释放")

    def detect_with_text_only(self, image_path, text_prompt):
        """
        仅使用文本提示进行检测（无 bbox 提示）

        Args:
            image_path: 图像路径
            text_prompt: 文本提示（如 "car"、"vehicle"）

        Returns:
            list: 检测结果 [{'box': [left, top, width, height], 'confidence': float}, ...]
        """
        try:
            # 设置图像
            self.sam3_predictor.set_image(image_path)

            # 仅使用文本提示，不提供 bbox
            results = self.sam3_predictor(
                text=[text_prompt],
                bboxes=None,
                labels=None
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

                        # 过滤过小的检测框（面积 < 100 像素²）
                        if width * height < 100:
                            continue

                        # 过滤低置信度
                        if conf < self.confidence_threshold:
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

    def detect_with_bboxes(self, image_path, text_prompt, bboxes):
        """
        使用文本提示 + bbox 提示进行检测

        Args:
            image_path: 图像路径
            text_prompt: 文本提示
            bboxes: bbox 列表 [[x_min, y_min, x_max, y_max], ...]

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

    def visualize_results(self, image_path, detections, output_path):
        """
        可视化检测结果

        Args:
            image_path: 原始图像路径
            detections: 检测结果列表
            output_path: 输出路径
        """
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # 尝试加载字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            except:
                font = ImageFont.load_default()

            # 绘制检测结果
            colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
            for idx, det in enumerate(detections):
                box = det['box']
                conf = det['confidence']

                left, top, width, height = box
                x1, y1 = left, top
                x2, y2 = left + width, top + height

                color = colors[idx % len(colors)]

                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # 绘制置信度标签
                label = f"{conf:.2f}"
                text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1 - 30), label, fill="white", font=font)

            # 保存
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            img.save(output_path)
            print(f"可视化结果保存到: {output_path}")

        except Exception as e:
            import traceback
            print(f"可视化失败: {e}")
            print(traceback.format_exc())

    def save_cropped_detections(self, image_path, detections, output_dir):
        """
        保存检测框裁剪的图像到指定文件夹

        Args:
            image_path: 原始图像路径
            detections: 检测结果列表
            output_dir: 输出文件夹路径
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 加载原始图像
            img = Image.open(image_path).convert("RGB")

            # 保存每个检测框的裁剪图像
            saved_count = 0
            for idx, det in enumerate(detections):
                box = det['box']
                conf = det['confidence']

                left, top, width, height = box
                x1 = int(left)
                y1 = int(top)
                x2 = int(left + width)
                y2 = int(top + height)

                # 确保坐标在图像范围内
                img_width, img_height = img.size
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                # 裁剪检测框区域
                cropped_img = img.crop((x1, y1, x2, y2))

                # 生成文件名
                filename = f"detection_{idx + 1}_conf{conf:.2f}.jpg"
                filepath = os.path.join(output_dir, filename)

                # 保存裁剪图像
                cropped_img.save(filepath, 'JPEG')
                saved_count += 1
                print(f"  保存检测框 {idx + 1}: {filepath} (置信度: {conf:.2f})")

            print(f"\n共保存 {saved_count} 个检测框裁剪图像到: {output_dir}")
            return saved_count

        except Exception as e:
            import traceback
            print(f"保存检测框图像失败: {e}")
            print(traceback.format_exc())
            return 0

    def run(self, image_path, text_prompt, output_path="sam3_result.jpg", bboxes=None, crop_output_dir=None):
        """
        完整的 SAM3 检测流程

        Args:
            image_path: 待检测图像路径
            text_prompt: 文本提示（如 "car or vehicle"）
            output_path: 可视化结果输出路径
            bboxes: 可选的 bbox 提示列表，如果为 None 则仅使用文本提示
            crop_output_dir: 可选的检测框裁剪图像输出目录，如果为 None 则不保存

        Returns:
            list: 检测结果
        """
        print("=" * 80)
        print("开始 SAM3 检测流程")
        print("=" * 80)

        # 初始化 SAM3
        print("\n【步骤 1】初始化 SAM3...")
        self.init_sam3()

        # 执行检测
        print(f"\n【步骤 2】执行检测...")
        if bboxes:
            print(f"  使用文本提示 + {len(bboxes)} 个 bbox 提示")
            detections = self.detect_with_bboxes(image_path, text_prompt, bboxes)
        else:
            print(f"  仅使用文本提示: '{text_prompt}'")
            detections = self.detect_with_text_only(image_path, text_prompt)

        print(f"\n检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            print(f"  目标 {i + 1}: box={det['box']}, confidence={det['confidence']:.2f}")

        # 释放显存
        print("\n【步骤 3】释放显存...")
        self.release_sam3()

        # 可视化结果
        print("\n【步骤 4】生成可视化结果...")
        self.visualize_results(image_path, detections, output_path)

        # 保存检测框裁剪图像
        if crop_output_dir:
            print("\n【步骤 5】保存检测框裁剪图像...")
            self.save_cropped_detections(image_path, detections, crop_output_dir)

        print("\n" + "=" * 80)
        print("检测流程完成！")
        print("=" * 80)

        return detections


def main():
    """主函数"""
    # 配置参数
    SAM3_MODEL_PATH = "/home/sun/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"
    # IMAGE_PATH = "./assets/img00672.jpg"  # 修改为你的图像路径
    IMAGE_PATH = "./ultralytics/assets/bus.jpg"  # 修改为你的图像路径
    IMAGE_PATH = "/home/sun/data/coding/object_detection_based_on_context_learning/UA-DETRAC/DETRAC-Images/MVI_39361/img01794.jpg"  # 修改为你的图像路径
    IMAGE_PATH = "target3.png"  # 修改为你的图像路径
    TEXT_PROMPT = "vehicle"  # 修改为你的文本提示
    TEXT_PROMPT = "collapse"  # 修改为你的文本提示
    # TEXT_PROMPT = "person"  # 修改为你的文本提示
    OUTPUT_PATH = "./sam3_detection_result.jpg"  # 输出路径
    CROP_OUTPUT_DIR = "./detection_crops"  # 检测框裁剪图像输出目录

    # 创建检测器
    detector = SAM3OnlyDetector(
        sam3_model_path=SAM3_MODEL_PATH,
        device='cuda',
        confidence_threshold=0.7
    )

    # 运行检测流程（仅使用文本提示）
    # results = detector.run(
    #     image_path=IMAGE_PATH,
    #     text_prompt=TEXT_PROMPT,
    #     output_path=OUTPUT_PATH,
    #     crop_output_dir=CROP_OUTPUT_DIR
    # )
    #
    # print(f"\n最终检测结果数量: {len(results)}")

    # 如果有手动标注的 bbox，可以使用以下方式：
    manual_bboxes = [
        [266, 299, 349, 326],  # [x_min, y_min, x_max, y_max]
        # [400, 200, 600, 400],
    ]
    results = detector.run(
        image_path=IMAGE_PATH,
        text_prompt=TEXT_PROMPT,
        output_path=OUTPUT_PATH,
        crop_output_dir=CROP_OUTPUT_DIR,
        bboxes=manual_bboxes
    )
    print(f"\n最终检测结果数量: {len(results)}")


if __name__ == '__main__':
    main()
