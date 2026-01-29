# 🚗 ALPR System for Non-Ideal Scenarios<br>针对现实生活非理想场景的车牌号识别系统

![alt text](https://img.shields.io/badge/license-MIT-blue.svg)
![alt text](https://img.shields.io/badge/python-3.8%2B-blue)
![alt text](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![alt text](https://img.shields.io/badge/YOLO-v8-green)

⚠️ **重要声明**: 本项目为课程作业开源，仅供学习参考

> 本项目是四川大学 2025-2026 学年秋季学期数字图像处理课程的期末结课大作业，该学期的期末大作业任务为：**完成车牌检测和车牌号的识别** 。欢迎需要完成类似课程任务的同学学习、参考和交流。

## 📖 项目介绍 (Introduction)

本项目针对现实生活中车牌识别面临的倾斜、模糊、光照复杂以及**多底色（蓝牌/绿牌）** 等非理想场景挑战，提出了一套基于深度学习与数字图像处理相结合的级联式车牌识别系统。
本项目结合了 YOLOv8-Pose 关键点检测与 LPRNet 端到端识别网络，并引入了基于 OTSU 二值化与 **智能反色** 的图像预处理方案，显著提升了模型在 CCPD 高难度数据集及现实测试集 **（由老师给定）** 上的识别准确率。

### 核心特性

- **高精度定位与矫正** ：利用 YOLOv8-Pose 回归车牌四角点，通过透视变换消除几何畸变。
- **鲁棒的图像预处理** ：设计了 "CLAHE + OTSU + 智能反色 + 连通域去噪" 的流水线，将蓝牌和绿牌统一为“黑底白字”特征，去除纹理干扰。
- **变长序列识别** ：使用 LPRNet + CTC Loss，无需字符分割即可同时处理 7 位（燃油车）和 8 位（新能源）车牌。
- 优异的实测性能：在非理想场景测试集上，端到端识别准确率达到 **74.26%** 。

## 🛠️ 技术路线 (Methodology)

本项目的推理流程遵循“检测 - 矫正 - 增强 - 识别”的级联架构：

![alt text](method_graph.png)

1. 车牌检测与关键点定位 (YOLOv8-Pose)
   采用改进的 YOLOv8n-pose 模型。不同于普通的目标检测，该模型引入了关键点回归分支：
   输入：原始高分辨率图像。
   输出：车牌置信度、边界框、四个角点坐标（左上、右上、右下、左下）。
   优势：Anchor-Free 机制适应不同长宽比车牌；Pose 损失函数确保了角点回归的精准度，为后续几何矫正奠定基础。
2. 几何畸变矫正
   利用预测的四个关键点计算单应性矩阵，对车牌区域执行透视变换，将任意角度倾斜的车牌“拉直”为 188×48 的标准正视图。
3. 结构化特征提取

![alt text](step_process.png)

这是本项目提升识别率的关键步骤。为了克服光照和底色差异，采用如下数字图像处理方案：

- 高分辨率输入：保留汉字笔画细节。
- CLAHE：限制对比度自适应直方图均衡化，均衡局部光照。
- OTSU 二值化：自适应阈值分割，剥离颜色与背景纹理。
- 智能反色：自动检测 ROI 区域（图像中心）的像素密度。若白色占比 > 50%（判定为白底黑字），则执行按位取反。将蓝牌（黑底白字）和绿牌（白底黑字）统一转化为黑底白字输入给识别网络。
- 连通域去噪：滤除面积小于阈值的孤立噪点（如雪花噪声）。

4. 字符序列识别 (LPRNet)
   模型：轻量级 LPRNet。
   损失函数：CTC Loss，解决字符对齐和长短不一的问题。
   解码：CTC Greedy Decoding。

## 📊 实验结果 (Results)

实验基于从 CCPD (Chinese City Parking Dataset) 随机抽取的 50000 张图像数据集进行训练，并在包含大量倾斜、模糊、阴影的非理想测试集上进行了验证（从 CCPD 数据集随机抽取的 10000 张图像）。

- <font color="hotpink">YOLOv8 检出率: 98.02%</font>
- <font color="hotpink">LPRNet 识别准确率: 75.76%</font>
- <font color="hotpink">端到端系统准确率: 74.26%</font>

通过剥离颜色和纹理信息，强制网络专注于字符的拓扑结构，极大降低了过拟合风险，显著缩小了训练集与测试集之间的性能鸿沟。

该方案能够正确识别大角度倾斜、模糊以及不同类型的车牌。

![alt text](image.png)

## 🚀 快速开始 (Quick Start)

### 环境依赖

本次用到了如下依赖

```Bash
ultralytics
torch
opencv-python
numpy
```

### 数据准备

请从官方渠道下载 [CCPD 数据集](https://github.com/detectRecog/CCPD)，并运行 data/ 目录下的脚本进行数据集预处理（随机选择 50000 张图像作为训练集，10000 张图像作为验证集，并转换为 YOLO 格式及 LPRNet 格式）。

```bash
cd ./dataset
python ./preprocess.py
```

### 训练

1. 训练检测模型 (YOLOv8-Pose)
   进入项目根目录下的 ./main 目录，运行训练 YOLOv8 的脚本。

```bash
cd ../
cd ./main

python train_yolov8.py --epochs <你希望训练的轮数> --batch-size <你的训练设备能承受的批次大小>
# 其他参数自行阅读代码，按照自己的需求设置
```

#### 训练识别模型 (LPRNet)

在刚才的目录下，继续运行训练 LPRNet 的脚本。

```bash
python train_lprnet.py --max_epoch <你希望训练的轮数>
# 其他参数按照自己的需要设置
```

### 推理

使用预训练模型对单张图片识别：

```bash
# 脚本在 ./main 下
python ./process_single_samples.py --image_path <你的图像路径> --img_size <图像大小> --save_dir <你希望保存结果的目录>
```

使用预训练模型对老师给定测试集进行识别：

```bash
# 脚本在 ./main 下
python ./evaluate.py
```

# 📂 项目结构

```bash
ALPR/
├── dataset/ # 数据集处理脚本
├── models/ # LPRNet 模型定义
├── main/ # 数据集处理脚本
├── .gitignore
├── simhei.ttf # 中文字体（用于可视化）
└── README.md
```

# 📝 引用与致谢

感谢 CCPD 数据集团队提供的高质量开源数据。

感谢本课程任课老师的悉心指导。

YOLOv8 实现基于 Ultralytics。

LPRNet 部分参考了 [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)。

。
