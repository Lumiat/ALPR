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
        稳健版预处理：
        灰度 -> CLAHE增强 -> OTSU -> 基于中心区域密度的反色 -> 连通域去噪
        """
        
        # === 1. 灰度化 (保留你的加权公式，效果更好) ===
        if len(img.shape) == 3:
            B, G, R = cv2.split(img.astype(np.float32))
            gray = 0.30 * R + 0.59 * G + 0.11 * B
            gray = gray.astype(np.uint8)
        else:
            gray = img

        # === 3. CLAHE 光线均衡化 ===
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # === 4. OTSU 二值化 ===
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # === 5. 极稳健的反色逻辑 (基于面积占比) ===
        # 核心假设：【字符面积】 < 【背景面积】
        # 做法：截取图片中心区域（避开边框干扰），统计白色像素占比。
        h, w = binary.shape
        # 截取中心 ROI: 高度 20%~80%, 宽度 10%~90%
        # 这个区域包含了大部分字符和背景，且避开了边框
        roi = binary[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        
        white_pixel_ratio = np.count_nonzero(roi == 255) / roi.size
        
        # 判定：
        # 如果 ROI 大部分 (>50%) 是白色 -> 说明背景是白的 (绿牌/白牌) -> 需要反色
        # 如果 ROI 大部分是黑色 -> 说明背景是黑的 (蓝牌/黑牌) -> 保持不变
        if white_pixel_ratio > 0.6:
            binary = cv2.bitwise_not(binary)

        # === 6. 连通域去噪 (去除雪花) ===
        # 现在我们可以确信：背景是黑的(0)，字符和噪点是白的(255)
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        clean_binary = np.zeros_like(binary)
        
        for i in range(1, num_labels): 
            area = stats[i, cv2.CC_STAT_AREA]
            # 滤除微小噪点 (小于15像素)
            if area > 20: 
                clean_binary[labels_map == i] = 255

        # === 保存查看 ===
        if save_name is not None:
            save_path = os.path.join(SAVE_PROCESS_DIR, save_name)
            cv2.imwrite(save_path, clean_binary)

        # === 转回网络输入格式 ===
        img_3c = cv2.cvtColor(clean_binary, cv2.COLOR_GRAY2BGR)
        img_3c = img_3c.astype('float32')
        img_3c -= 127.5
        img_3c *= 0.0078125
        img_3c = np.transpose(img_3c, (2, 0, 1))
        
        return img_3c
