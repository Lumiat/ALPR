# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

# ================= 1. 环境与路径配置 =================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# 导入自定义模块
from dataset.load_data import CHARS, CHARS_DICT, LPRDataLoader 
from dataset.preprocess import rectification
from model.LPRNet import build_lprnet

TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
TEST_LABEL_PATH = os.path.join(PROJECT_ROOT, "dataset", "test", "label.txt")
SAVE_DIR = os.path.join(PROJECT_ROOT, "result", "prediction")
DEBUG_CROP_DIR = os.path.join(PROJECT_ROOT, "dataset", "result", "debug_crops")

YOLO_WEIGHT = os.path.join(PROJECT_ROOT, "runs", "ccpd_pose_train", "weights", "best.pt")
LPR_WEIGHT = os.path.join(PROJECT_ROOT, "weight", "Final_LPRNet_model.pth")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DEBUG_CROP_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================= 2. 辅助函数 =================
def decode_lprnet(preds):
    preds = preds.cpu().detach().numpy() 
    pred_labels = []
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        preds_idx = np.argmax(pred, axis=0) 
        no_repeat_blank_label = []
        pre_c = preds_idx[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preds_idx:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_str = "".join([CHARS[idx] for idx in no_repeat_blank_label])
        pred_labels.append(pred_str)
    return pred_labels[0]

def draw_result(img, box, text, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_size = 80
    font_path = os.path.join(PROJECT_ROOT, "simhei.ttf")
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    text_pos = (x1, max(0, y1 - font_size - 10))
    draw.text(text_pos, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ================= 3. 主评估逻辑 =================

def evaluate():
    print(f"Loading Models...")
    yolo_model = YOLO(YOLO_WEIGHT)

    lpr_net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    lpr_net.load_state_dict(torch.load(LPR_WEIGHT, map_location=DEVICE))
    lpr_net.to(DEVICE)
    lpr_net.eval()

    print("Initializing Preprocessor from load_data.py...")
    preprocessor = LPRDataLoader(img_dirs=[], imgSize=[94, 24], lpr_max_len=8)

    if not os.path.exists(TEST_LABEL_PATH):
        raise FileNotFoundError(f"Label file not found at {TEST_LABEL_PATH}")
    
    with open(TEST_LABEL_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    detected_count = 0
    correct_count = 0
    
    print(f"Start Evaluation on {total_samples} images...")
    
    for line in tqdm(lines):
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 2: continue
        img_name, gt_plate = parts[0], parts[1]
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        # === A. YOLO 检测 ===
        results = yolo_model(img, imgsz=960, conf=0.25, verbose=False)
        
        if len(results[0].boxes) == 0:
            cv2.imwrite(os.path.join(SAVE_DIR, "FAIL_" + img_name), img)
            continue
            
        detected_count += 1
        best_idx = torch.argmax(results[0].boxes.conf).item()
        box = results[0].boxes.xyxy[best_idx].cpu().numpy()
        keypoints = results[0].keypoints.xy[best_idx].cpu().numpy()
        
        # === B. 图像矫正 ===
        # 这里使用的是经过提亮后的 img，所以即便原图很黑，现在也能截出清晰的车牌
        rectified_crop = rectification(img, keypoints)
        
        # === C. LPRNet 识别 ===
        # 调用 load_data.py 中最新的 transform
        processed_img_numpy = preprocessor.transform(rectified_crop, save_name=None)
        
        lpr_input = torch.from_numpy(processed_img_numpy).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = lpr_net(lpr_input)
            pred_plate = decode_lprnet(preds)
            
        # === D. 统计与可视化 ===
        is_correct = (pred_plate == gt_plate)
        if is_correct:
            correct_count += 1
        
        vis_img = draw_result(img.copy(), box, f"{pred_plate}", color=(0, 255, 0) if is_correct else (0, 0, 255))
        save_name = ("OK_" if is_correct else "ERR_") + img_name
        cv2.imwrite(os.path.join(SAVE_DIR, save_name), vis_img)

        # === E. [DEBUG] 保存矫正后的裁剪图 ===
        debug_crop_name = f"{gt_plate}_Pred({pred_plate})_{img_name}"
        cv2.imwrite(os.path.join(DEBUG_CROP_DIR, debug_crop_name), rectified_crop)

    # 输出报告
    detection_rate = detected_count / total_samples * 100
    recognition_acc = correct_count / detected_count * 100 if detected_count > 0 else 0
    end_to_end_acc = correct_count / total_samples * 100

    print("\n" + "="*50)
    print("【车牌识别系统最终评估报告】")
    print(f"测试集样本数: {total_samples}")
    print(f"YOLOv8 检出率: {detection_rate:.2f}%")
    print("-" * 30)
    print(f"LPRNet 识别准确率 (基于已检出): {recognition_acc:.2f}%")
    print("-" * 30)
    print(f"系统端到端准确率: {end_to_end_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate()
