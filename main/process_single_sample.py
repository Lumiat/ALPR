# -*- coding: utf-8 -*-
"""
单张图像预处理脚本
功能：使用 LPRDataLoader 的预处理流程处理指定图像，并可视化结果
"""
import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 路径处理
try:
    root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("ALPR")+1])
    sys.path.append(root)
    os.chdir(root)
except:
    pass

from dataset.load_data import CHARS, CHARS_DICT

# ===== 修复：正确获取项目根目录 =====
# 脚本在 ./main/ 下，需要向上一级到达项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # /research-intern02/xjy/ALPR/main
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # /research-intern02/xjy/ALPR
FONT_PATH = os.path.join(PROJECT_ROOT, "simhei.ttf")

# 验证字体文件是否存在
if not os.path.exists(FONT_PATH):
    print(f"Warning: Font file not found at {FONT_PATH}")
    print(f"Using default font instead.")
    chinese_font = None
else:
    chinese_font = FontProperties(fname=FONT_PATH)

class SingleImageProcessor:
    """单张图像预处理器"""
    
    def __init__(self, img_size=[94, 24]):
        self.img_size = img_size
    
    def transform(self, img):
        """
        预处理流程（与 LPRDataLoader 完全一致）
        1. 灰度化
        2. CLAHE 光线均衡化
        3. OTSU 二值化
        4. 智能反色
        5. 去除小连通块
        """
        
        # 1. 灰度化
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2. CLAHE 光线均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 3. OTSU 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. 智能反色
        h, w = binary.shape
        roi = binary[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        white_ratio = np.count_nonzero(roi == 255) / roi.size
        
        if white_ratio > 0.5:
            binary = cv2.bitwise_not(binary)

        # 5. 去除小连通块
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean_binary = np.zeros_like(binary)
        
        for i in range(1, num_labels): 
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 15: 
                clean_binary[labels_map == i] = 255

        return gray, enhanced, binary, clean_binary
    
    def preprocess_for_network(self, img):
        """转换为网络输入格式"""
        # 复制为3通道
        img_3c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 归一化
        img_3c = img_3c.astype('float32')
        img_3c -= 127.5
        img_3c *= 0.0078125
        img_3c = np.transpose(img_3c, (2, 0, 1))
        return img_3c
    
    def process_image(self, image_path, save_dir="./result"):
        """
        处理单张图像并保存可视化结果
        
        Args:
            image_path: 输入图像路径
            save_dir: 保存结果的目录
        """
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to read image from {image_path}")
            return
        
        # 调整尺寸
        height, width, _ = img.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            img_resized = cv2.resize(img, self.img_size)
        else:
            img_resized = img.copy()
        
        # 预处理
        gray, enhanced, binary, clean_binary = self.transform(img_resized)
        
        # 提取标签信息（如果文件名包含车牌号）
        basename = os.path.basename(image_path)
        imgname, _ = os.path.splitext(basename)
        plate_str = imgname.split("_")[0] if "_" in imgname else "Unknown"
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 可视化并保存
        self._visualize_process(img, img_resized, gray, enhanced, binary, clean_binary, 
                                plate_str, save_dir, basename)
        
        # 保存最终处理结果
        final_save_path = os.path.join(save_dir, f"processed_{basename}")
        cv2.imwrite(final_save_path, clean_binary)
        
        print(f"✓ Processing complete!")
        print(f"  - Plate Number: {plate_str}")
        print(f"  - Original Size: {img.shape[:2]}")
        print(f"  - Resized to: {img_resized.shape[:2]}")
        print(f"  - Visualization saved to: {os.path.join(save_dir, f'process_steps_{basename}')}")
        print(f"  - Processed image saved to: {final_save_path}")
    
    def _visualize_process(self, original, resized, gray, enhanced, binary, clean_binary, 
                          plate_str, save_dir, basename):
        """可视化预处理的各个步骤"""
        
        # 转换 BGR 到 RGB 用于显示
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # ===== 修改：根据字体可用性设置标题 =====
        if chinese_font is not None:
            fig.suptitle(f'车牌预处理流程 - {plate_str}', fontproperties=chinese_font, fontsize=16)
        else:
            fig.suptitle(f'License Plate Preprocessing - {plate_str}', fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(original_rgb)
        if chinese_font is not None:
            axes[0, 0].set_title('1. 原始图像', fontproperties=chinese_font)
        else:
            axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')
        
        # 调整尺寸后
        axes[0, 1].imshow(resized_rgb)
        if chinese_font is not None:
            axes[0, 1].set_title(f'2. 调整尺寸 ({self.img_size[0]}x{self.img_size[1]})', 
                                fontproperties=chinese_font)
        else:
            axes[0, 1].set_title(f'2. Resized ({self.img_size[0]}x{self.img_size[1]})')
        axes[0, 1].axis('off')
        
        # 灰度化
        axes[0, 2].imshow(gray, cmap='gray')
        if chinese_font is not None:
            axes[0, 2].set_title('3. 灰度化', fontproperties=chinese_font)
        else:
            axes[0, 2].set_title('3. Grayscale')
        axes[0, 2].axis('off')
        
        # CLAHE 增强
        axes[1, 0].imshow(enhanced, cmap='gray')
        if chinese_font is not None:
            axes[1, 0].set_title('4. CLAHE 光线均衡', fontproperties=chinese_font)
        else:
            axes[1, 0].set_title('4. CLAHE Enhancement')
        axes[1, 0].axis('off')
        
        # OTSU 二值化
        axes[1, 1].imshow(binary, cmap='gray')
        if chinese_font is not None:
            axes[1, 1].set_title('5. OTSU 二值化 + 反色', fontproperties=chinese_font)
        else:
            axes[1, 1].set_title('5. OTSU Binarization')
        axes[1, 1].axis('off')
        
        # 去噪后
        axes[1, 2].imshow(clean_binary, cmap='gray')
        if chinese_font is not None:
            axes[1, 2].set_title('6. 去除小连通块（最终）', fontproperties=chinese_font)
        else:
            axes[1, 2].set_title('6. Noise Removal (Final)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        save_path = os.path.join(save_dir, f'process_steps_{basename}')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='预处理单张车牌图像')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='输入图像路径')
    parser.add_argument('--img_size', type=int, nargs=2, default=[94, 24],
                       help='目标图像尺寸 [width, height]')
    parser.add_argument('--save_dir', type=str, default='./result/single_image_process',
                       help='保存结果的目录')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = SingleImageProcessor(img_size=args.img_size)
    
    # 处理图像
    processor.process_image(args.image_path, args.save_dir)


if __name__ == "__main__":
    main()
