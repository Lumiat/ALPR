from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os

# ================= 1. 构建符合 CCPD 的字符集 =================
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHANUMS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
CHARS = PROVINCES + ALPHANUMS + ['-']
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

# 保存路径
SAVE_PROCESS_DIR = "/research-intern02/xjy/ALPR/dataset/ccpd_lprnet_processes"

class LPRDataLoader(Dataset):
    def __init__(self, img_dirs, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dirs = img_dirs
        self.img_paths = []
        
        for img_dir in img_dirs:
            for root, _, files in os.walk(img_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.img_paths.append(os.path.join(root, file))
        
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        
        if not os.path.exists(SAVE_PROCESS_DIR):
            os.makedirs(SAVE_PROCESS_DIR)
        
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        
        if Image is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
            
        basename = os.path.basename(filename)
        Image = self.transform(Image, save_name=basename)

        imgname, _ = os.path.splitext(basename)
        plate_str = imgname.split("_")[0] 
        
        label = list()
        for c in plate_str:
            if c in CHARS_DICT:
                label.append(CHARS_DICT[c])
            else:
                pass

        return Image, label, len(label)

    def transform(self, img, save_name=None):
        """
        清晰化预处理流程 (适配 188x48):
        1. 灰度化
        2. Gamma 提亮 (针对暗图)
        3. CLAHE 光线均衡化
        4. OTSU 二值化
        5. 智能反色 (基于中心区域密度)
        6. 去除小连通块 (去噪)
        """
        
        # === 1. 灰度化 ===
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # === 2. Gamma 提亮 ===
        # 避免暗部车牌在二值化时丢失
        # mean_brightness = np.mean(gray)
        # if mean_brightness < 80: 
        #     gamma = 0.6 # 提亮系数
        #     lookUpTable = np.empty((1, 256), np.uint8)
        #     for i in range(256):
        #         lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        #     gray = cv2.LUT(gray, lookUpTable)

        # === 3. CLAHE 光线均衡化 ===
        # 增强局部对比度，让模糊的汉字笔画显现出来
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # === 4. OTSU 二值化 ===
        # 自动计算阈值，不使用 Adaptive Threshold 以避免笔画断裂
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # === 5. 智能反色 (基于中心 ROI 占比) ===
        # 目的：统一转换为【黑底白字】
        h, w = binary.shape
        # 截取中心 60% 区域，避开边框干扰
        roi = binary[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        # 统计白色像素占比
        white_ratio = np.count_nonzero(roi == 255) / roi.size
        
        # 逻辑：
        # 如果白色占比 > 50% -> 说明背景是白的 (绿牌) -> 反色
        # 如果白色占比 < 50% -> 说明背景是黑的 (蓝牌) -> 不变
        if white_ratio > 0.5:
            binary = cv2.bitwise_not(binary)

        # === 6. 去除小连通块 (去噪) ===
        # 使用 connectedComponents 物理删除噪点
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        clean_binary = np.zeros_like(binary)
        
        for i in range(1, num_labels): 
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 【关键参数适配】：
            # 图像分辨率从 94x24 翻倍到 188x48，面积翻了4倍。
            # 原来阈值是 15，现在理论上是 60。
            # 这里设置 50，稍微保守一点，防止误删断裂的汉字笔画。
            if area > 15: 
                clean_binary[labels_map == i] = 255

        # === 保存查看 (建议在训练前检查这个文件夹) ===
        if save_name is not None:
            save_path = os.path.join(SAVE_PROCESS_DIR, save_name)
            cv2.imwrite(save_path, clean_binary)

        # === 转回网络输入格式 ===
        # 复制为3通道
        img_3c = cv2.cvtColor(clean_binary, cv2.COLOR_GRAY2BGR)
        # 归一化
        img_3c = img_3c.astype('float32')
        img_3c -= 127.5
        img_3c *= 0.0078125
        img_3c = np.transpose(img_3c, (2, 0, 1))
        
        return img_3c
