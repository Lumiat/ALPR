from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os

# ================= 1. 构建符合 CCPD 的字符集 =================
# 省份
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
# 字母数字 (合并 alphabets 和 ads，去重并排序，保留 'O')
ALPHANUMS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# 最终字符表: 省份 + 字母数字 + Blank符号('-')
# 注意: Blank 符号必须放在最后，因为 CTCLoss 默认 blank=len(CHARS)-1
CHARS = PROVINCES + ALPHANUMS + ['-']

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dirs, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dirs = img_dirs
        self.img_paths = []
        
        # 递归搜索所有图片，替代 imutils
        for img_dir in img_dirs:
            for root, _, files in os.walk(img_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.img_paths.append(os.path.join(root, file))
        
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        
        # 异常处理：防止读取损坏图片
        if Image is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        # ================= 2. 核心修改：文件名解析 =================
        # 你的文件名格式: "川A0XN95_2397.jpg"
        # 我们只需要 "_" 前面的部分
        imgname, _ = os.path.splitext(basename)
        plate_str = imgname.split("_")[0] 
        
        label = list()
        for c in plate_str:
            # 兼容性处理：如果遇到字典里没有的字符，跳过或报错
            if c in CHARS_DICT:
                label.append(CHARS_DICT[c])
            else:
                print(f"Warning: Character {c} not in dict (File: {basename})")

        return Image, label, len(label)

    def transform(self, img):
        try:
            # 转换到 YUV 颜色空间，只对亮度通道 Y 做均衡化
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        except:
            pass # 防止极端情况报错
    
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img
