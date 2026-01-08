import os
import cv2
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 请将此处修改为你本地 CCPD 数据集的根目录路径
# 该路径下应包含 ccpd_base, ccpd_green, ccpd_blur 等子文件夹
CCPD_ROOT_PATH = r"./CCPD"  

# 输出路径配置
OUTPUT_ROOT = "./dataset"
YOLO_DIR = os.path.join(OUTPUT_ROOT, "ccpd_yolo")
LPR_DIR = os.path.join(OUTPUT_ROOT, "ccpd_lprnet")

# 数据量限制
TRAIN_LIMIT = 50000
VAL_LIMIT = 10000
TOTAL_LIMIT = TRAIN_LIMIT + VAL_LIMIT

# 验证集比例 (虽然有固定数量，但为了打散数据，我们用 random 决定)
# 这里通过计数器强制控制数量，split_ratio 仅作参考
# ===========================================

# CCPD 字符映射表 (完全依照提供的规范)
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def ensure_dirs():
    """创建必要的文件夹结构"""
    # YOLO 目录
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', subset), exist_ok=True)
        
    # LPRNet 目录
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(LPR_DIR, subset), exist_ok=True)

def parse_filename(filename):
    """
    解析 CCPD 文件名
    格式示例: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    """
    try:
        name_split = filename.split('-')
        
        # 1. 边界框 (Field 2): 154&383_386&473 -> xmin&ymin_xmax&ymax
        bbox_str = name_split[2]
        top_left_str, bottom_right_str = bbox_str.split('_')
        xmin, ymin = map(int, top_left_str.split('&'))
        xmax, ymax = map(int, bottom_right_str.split('&'))
        bbox = [xmin, ymin, xmax, ymax]
        
        # 2. 关键点 (Field 3): 386&473_177&454_154&383_363&402
        # CCPD 顺序: 右下(BR), 左下(BL), 左上(TL), 右上(TR)
        kps_str = name_split[3].split('_')
        kps_raw = [list(map(int, pt.split('&'))) for pt in kps_str]
        
        # 转换为我们需要的逻辑顺序: 左上(TL), 右上(TR), 右下(BR), 左下(BL)
        # raw[2]=TL, raw[3]=TR, raw[0]=BR, raw[1]=BL
        landmarks = [kps_raw[2], kps_raw[3], kps_raw[0], kps_raw[1]]
        
        # 3. 车牌号 (Field 4): 0_0_22_27_27_33_16
        label_str = name_split[4]
        indexes = list(map(int, label_str.split('_')))
        
        # 映射字符
        # 第1位: 省份
        plate_chars = PROVINCES[indexes[0]]
        # 第2位: 字母
        plate_chars += ALPHABETS[indexes[1]]
        # 后续位: 字母或数字
        for idx in indexes[2:]:
            # 如果是 'O' (作为无字符占位符)，通常跳过，但在CCPD中通常是有效位
            # 此处直接拼接
            plate_chars += ADS[idx]
            
        return bbox, landmarks, plate_chars
        
    except Exception as e:
        # 文件名格式不符或损坏
        return None, None, None

def rectification(img, landmarks):
    """透视变换矫正"""
    # 源点: TL, TR, BR, BL
    src_pts = np.float32(landmarks)
    
    # 目标尺寸 (LPRNet 常用输入)
    target_w, target_h = 94, 24
    
    # 目标点: TL, TR, BR, BL
    dst_pts = np.float32([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (target_w, target_h))
    return warped

def main():
    print(f"开始扫描数据集: {CCPD_ROOT_PATH}")
    
    # 递归搜索所有 jpg 图片
    all_images = list(Path(CCPD_ROOT_PATH).rglob("*.jpg"))
    print(f"找到图片总数: {len(all_images)}")
    
    if len(all_images) < TOTAL_LIMIT:
        print(f"警告: 图片总数 ({len(all_images)}) 少于需求 ({TOTAL_LIMIT})，将使用全部图片。")
    
    # 随机打乱以保证分布均匀
    random.shuffle(all_images)
    
    ensure_dirs()
    
    train_count = 0
    val_count = 0
    
    pbar = tqdm(total=min(len(all_images), TOTAL_LIMIT))
    
    for img_path in all_images:
        # 检查是否已达到目标数量
        if train_count >= TRAIN_LIMIT and val_count >= VAL_LIMIT:
            break
            
        # 决定是训练集还是验证集
        if train_count < TRAIN_LIMIT:
            subset = 'train'
        elif val_count < VAL_LIMIT:
            subset = 'val'
        else:
            continue # 应该不会走到这里
            
        filename = img_path.name
        bbox, landmarks, plate_number = parse_filename(filename)
        
        if bbox is None:
            continue
            
        # 读取图片 (需要图片尺寸来归一化 YOLO 坐标)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # -----------------------------
        # 1. 处理 YOLO 数据 (使用原图)
        # -----------------------------
        
        # 1.1 复制图片
        dst_img_name = f"{plate_number}_{train_count+val_count}.jpg" # 重命名简单点，防止文件名过长
        dst_img_path = os.path.join(YOLO_DIR, 'images', subset, dst_img_name)
        shutil.copy(str(img_path), dst_img_path)
        
        # 1.2 生成标签 (归一化 xywh + 关键点)
        # YOLO center x, y, w, h
        dw = 1.0 / w
        dh = 1.0 / h
        
        x_center = ((bbox[0] + bbox[2]) / 2.0) * dw
        y_center = ((bbox[1] + bbox[3]) / 2.0) * dh
        width = (bbox[2] - bbox[0]) * dw
        height = (bbox[3] - bbox[1]) * dh
        
        # 关键点格式: x y visibility (2=visible)
        kps_str = []
        for pt in landmarks:
            kps_str.append(f"{pt[0]*dw:.6f} {pt[1]*dh:.6f} 2")
            
        # 写入 txt
        label_name = dst_img_name.replace('.jpg', '.txt')
        dst_label_path = os.path.join(YOLO_DIR, 'labels', subset, label_name)
        
        with open(dst_label_path, 'w') as f:
            # class_id=0, bbox, keypoints
            line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {' '.join(kps_str)}"
            f.write(line)
            
        # -----------------------------
        # 2. 处理 LPRNet 数据 (使用矫正图)
        # -----------------------------
        
        # 2.1 透视变换
        warped_img = rectification(img, landmarks)
        
        # 2.2 保存 (文件名为: 车牌号_ID.jpg)
        # 增加ID是为了避免车牌号重复导致文件覆盖
        # 对于 LPRNet 的 dataset loader，通常需要自己写解析逻辑去 split('_')[0]
        lpr_filename = f"{plate_number}_{train_count+val_count}.jpg"
        lpr_path = os.path.join(LPR_DIR, subset, lpr_filename)
        cv2.imwrite(lpr_path, warped_img)
        
        # 更新计数器
        if subset == 'train':
            train_count += 1
        else:
            val_count += 1
            
        pbar.update(1)
        pbar.set_description(f"Train: {train_count}/{TRAIN_LIMIT} | Val: {val_count}/{VAL_LIMIT}")

    pbar.close()
    print("\n预处理完成！")
    print(f"YOLO 数据集位于: {YOLO_DIR}")
    print(f"LPRNet 数据集位于: {LPR_DIR}")

if __name__ == "__main__":
    main()