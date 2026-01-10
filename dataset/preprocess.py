import os
import cv2
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区域 =================
CCPD_ROOT_PATH = r"./CCPD"  # 请修改为你实际的路径

OUTPUT_ROOT = "./dataset"
YOLO_DIR = os.path.join(OUTPUT_ROOT, "ccpd_yolo")
LPR_DIR = os.path.join(OUTPUT_ROOT, "ccpd_lprnet_balanced")

# 目标数据量
TARGET_TRAIN_NUM = 50000
TARGET_VAL_NUM = 10000
TOTAL_TARGET = TARGET_TRAIN_NUM + TARGET_VAL_NUM

# ===========================================

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def ensure_dirs():
    """创建文件夹，如果已存在则清空(慎用)或者直接覆盖"""
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', subset), exist_ok=True)
        os.makedirs(os.path.join(LPR_DIR, subset), exist_ok=True)

def parse_filename(filename):
    """解析文件名"""
    try:
        name_split = filename.split('-')
        bbox_str = name_split[2]
        top_left_str, bottom_right_str = bbox_str.split('_')
        xmin, ymin = map(int, top_left_str.split('&'))
        xmax, ymax = map(int, bottom_right_str.split('&'))
        bbox = [xmin, ymin, xmax, ymax]
        
        kps_str = name_split[3].split('_')
        kps_raw = [list(map(int, pt.split('&'))) for pt in kps_str]
        landmarks = [kps_raw[2], kps_raw[3], kps_raw[0], kps_raw[1]]
        
        label_str = name_split[4]
        indexes = list(map(int, label_str.split('_')))
        plate_chars = PROVINCES[indexes[0]] + ALPHABETS[indexes[1]]
        for idx in indexes[2:]:
            plate_chars += ADS[idx]
            
        return bbox, landmarks, plate_chars
    except:
        return None, None, None

def rectification(img, landmarks):
    """
    透视变换矫正 - 高分辨率版
    """
    src_pts = np.float32(landmarks)
    
    # === 修改点：分辨率翻倍 (94*2 -> 188, 24*2 -> 48) ===
    # 更大的图像能保留更多汉字细节
    target_w, target_h = 188, 48 
    
    dst_pts = np.float32([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (target_w, target_h))
    return warped

def get_balanced_dataset(all_images_list):
    """
    核心算法：类别平衡采样
    1. 按省份分组
    2. 计算每个省份应采样的配额
    3. 采样并合并
    """
    print("正在分析数据分布...")
    grouped = defaultdict(list)
    
    # 1. 分组
    for img_path in tqdm(all_images_list, desc="Grouping"):
        fname = img_path.name
        # 快速解析省份索引，不需要完全解析文件名，提升速度
        try:
            # -area-tilt-bbox-landmarks-plate- 
            # plate字段: 0_0_22... 第一个数字是省份索引
            plate_part = fname.split('-')[4]
            province_idx = int(plate_part.split('_')[0])
            province_char = PROVINCES[province_idx]
            grouped[province_char].append(img_path)
        except:
            continue

    # 2. 计算配额
    # 理想情况下，每个省份应该有 TOTAL_TARGET / 34 张图片
    quota_per_province = TOTAL_TARGET // len(PROVINCES)
    print(f"理想单省份配额: {quota_per_province}")
    
    final_list = []
    
    # 3. 采样
    for prov, img_list in grouped.items():
        random.shuffle(img_list)
        count = len(img_list)
        
        if count > quota_per_province:
            # 如果该省份图片很多（如皖），截断
            selected = img_list[:quota_per_province]
        else:
            # 如果该省份图片很少（如藏），全取
            selected = img_list
            # 可选：如果实在太少，可以考虑复制（Oversampling），但这通常对训练提升有限
            
        final_list.extend(selected)
        # print(f"省份 {prov}: 原有 {count} -> 采样 {len(selected)}")

    # 如果截断导致总数不够 TOTAL_TARGET，我们再次从剩余的图片中随机补充
    current_count = len(final_list)
    if current_count < TOTAL_TARGET:
        needed = TOTAL_TARGET - current_count
        print(f"数据量不足，正在从剩余数据中补充 {needed} 张...")
        
        # 收集所有未被选中的图片
        remaining = []
        selected_set = set(final_list)
        for img_list in grouped.values():
            for img in img_list:
                if img not in selected_set:
                    remaining.append(img)
        
        if len(remaining) >= needed:
            random.shuffle(remaining)
            final_list.extend(remaining[:needed])
        else:
            print("警告：原始数据总量不足以填满目标数量！")
            final_list.extend(remaining)

    # 再次打乱
    random.shuffle(final_list)
    print(f"最终采样总数: {len(final_list)}")
    return final_list

def main():
    print(f"开始扫描数据集: {CCPD_ROOT_PATH}")
    # 使用 Path 对象递归搜索
    all_images = list(Path(CCPD_ROOT_PATH).rglob("*.jpg"))
    print(f"找到原始图片总数: {len(all_images)}")
    
    # 获取平衡后的文件列表
    balanced_images = get_balanced_dataset(all_images)
    
    ensure_dirs()
    
    # 划分训练集和验证集
    # 简单切分：前 50000 训练，后 10000 验证
    # 注意：如果 balanced_images 总数不够 60000，则按比例切分
    total_avail = len(balanced_images)
    
    if total_avail >= TOTAL_TARGET:
        train_imgs = balanced_images[:TARGET_TRAIN_NUM]
        val_imgs = balanced_images[TARGET_TRAIN_NUM:TOTAL_TARGET]
    else:
        # 数据不够时，按 5:1 比例分配
        split_idx = int(total_avail * (TARGET_TRAIN_NUM / TOTAL_TARGET))
        train_imgs = balanced_images[:split_idx]
        val_imgs = balanced_images[split_idx:]
        print(f"注意：实际数据量 ({total_avail}) 小于目标 ({TOTAL_TARGET})")

    # 处理函数
    def process_subset(image_list, subset_name):
        print(f"正在处理 {subset_name} 集...")
        count = 0
        for i, img_path in enumerate(tqdm(image_list)):
            filename = img_path.name
            bbox, landmarks, plate_number = parse_filename(filename)
            
            if bbox is None: continue
            
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            
            # # --- 1. YOLO 数据 (使用原图) ---
            # dst_img_name = f"{plate_number}_{i}.jpg"
            # dst_img_path = os.path.join(YOLO_DIR, 'images', subset_name, dst_img_name)
            # shutil.copy(str(img_path), dst_img_path)
            
            # # YOLO 标签
            # dw, dh = 1.0 / w, 1.0 / h
            # x_center = ((bbox[0] + bbox[2]) / 2.0) * dw
            # y_center = ((bbox[1] + bbox[3]) / 2.0) * dh
            # width = (bbox[2] - bbox[0]) * dw
            # height = (bbox[3] - bbox[1]) * dh
            
            # kps_str = []
            # for pt in landmarks:
            #     kps_str.append(f"{pt[0]*dw:.6f} {pt[1]*dh:.6f} 2")
            
            # label_name = dst_img_name.replace('.jpg', '.txt')
            # dst_label_path = os.path.join(YOLO_DIR, 'labels', subset_name, label_name)
            # with open(dst_label_path, 'w') as f:
            #     line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {' '.join(kps_str)}"
            #     f.write(line)
                
            # --- 2. LPRNet 数据 (使用高分辨率矫正图) ---
            warped_img = rectification(img, landmarks)
            
            lpr_filename = f"{plate_number}_{i}.jpg"
            lpr_path = os.path.join(LPR_DIR, subset_name, lpr_filename)
            cv2.imwrite(lpr_path, warped_img)
            
            count += 1
        print(f"{subset_name} 处理完成: {count} 张")

    process_subset(train_imgs, 'train')
    process_subset(val_imgs, 'val')

    print("\n预处理全部完成！")
    print(f"YOLO 数据集: {YOLO_DIR}")
    print(f"LPRNet 数据集 (分辨率 188x48): {LPR_DIR}")

if __name__ == "__main__":
    main()