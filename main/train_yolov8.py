import os
import shutil
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ================= 配置区域 =================
# 项目根目录 (假设脚本运行在 main/ 目录下，我们需要回退一级或者直接指定绝对路径)
# 建议在项目根目录下运行 python main/train_yolov8.py，这里使用相对路径处理
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据集路径
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ccpd_yolo")
# 结果保存路径
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
# 权重保存路径
WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weight")

# 训练超参数
EPOCHS = 50           # 训练轮数 (50轮对 YOLOv8 足够收敛)
BATCH_SIZE = 8       # 根据显存调整，显存大可以改 32 或 64
IMG_SIZE = 640        # 输入图片大小
MODEL_NAME = 'yolov8n-pose.pt' # 使用 Nano 版本，速度快。追求精度可用 yolov8s-pose.pt

def ensure_dirs():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

def create_yaml():
    """
    动态生成 YOLOv8 训练所需的 data.yaml 文件
    """
    yaml_path = os.path.join(PROJECT_ROOT, "dataset", "ccpd_plate.yaml")
    
    # 关键点配置: CCPD 有 4 个角点
    # kpt_shape: [4, 3] 表示 4个点，每个点有 (x, y, visibility) 3个值
    data_config = {
        'path': DATASET_DIR,       # 数据集根目录
        'train': 'images/train',   # 训练集图像 (相对于 path)
        'val': 'images/val',       # 验证集图像 (相对于 path)
        'names': {0: 'license_plate'}, # 类别名称
        'kpt_shape': [4, 3],       # 关键点形状设定
        'flip_idx': [1, 0, 3, 2]   # 数据增强翻转时，关键点的互换索引 (左上<->右上, 左下<->右下)
    }
    
    # 写入 yaml 文件
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
        
    print(f"数据配置文件已生成: {yaml_path}")
    return yaml_path

def visualize_metrics(csv_path):
    """
    读取训练日志并可视化保存
    """
    print("正在绘制训练指标可视化图表...")
    try:
        # 读取 YOLO 输出的 results.csv (注意：表头可能有空格)
        df = pd.read_csv(csv_path)
        # 去除列名两端的空格
        df.columns = [c.strip() for c in df.columns]
        
        # 定义需要绘制的指标
        metrics_map = {
            'train/box_loss': 'Box Loss (Training)',
            'train/pose_loss': 'Keypoint Loss (Training)',
            'metrics/mAP50(B)': 'mAP@0.5',
            'metrics/mAP50-95(B)': 'mAP@0.5:0.95'
        }
        
        for col, title in metrics_map.items():
            if col in df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(df['epoch'], df[col], marker='o', markersize=2, linestyle='-')
                plt.title(f'Training Metric: {title}')
                plt.xlabel('Epochs')
                plt.ylabel(title)
                plt.grid(True)
                
                # 保存文件名格式: {metric}_visualize.png
                # 将斜杠替换为下划线以便作为文件名
                safe_name = col.replace('/', '_').replace('(', '').replace(')', '')
                save_path = os.path.join(RESULT_DIR, f"{safe_name}_visualize.png")
                plt.savefig(save_path)
                plt.close()
                print(f"已保存: {save_path}")
            else:
                print(f"警告: 未在日志中找到指标 {col}")
                
    except Exception as e:
        print(f"可视化失败: {e}")

def main():
    ensure_dirs()
    
    # 1. 创建配置文件
    yaml_path = create_yaml()
    
    # 2. 加载模型 (YOLOv8-Pose)
    print(f"加载模型: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME) 
    
    # 3. 开始训练
    # project: 保存的主目录
    # name: 本次训练的任务名
    print("开始训练...")
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=os.path.join(PROJECT_ROOT, 'runs'),
        name='ccpd_pose_train',
        exist_ok=True,   # 如果存在则覆盖
        plots=True,      # 自动画图 (YOLO自带的)
        device=0         # 显卡索引，如果没有显卡请改为 'cpu'
    )
    
    # 4. 导出与处理结果
    train_run_dir = os.path.join(PROJECT_ROOT, 'runs', 'ccpd_pose_train')
    
    # 4.1 移动最佳权重
    best_weight_src = os.path.join(train_run_dir, 'weights', 'best.pt')
    target_weight_dst = os.path.join(WEIGHT_DIR, 'yolov8.pth')
    
    if os.path.exists(best_weight_src):
        shutil.copy(best_weight_src, target_weight_dst)
        print(f"最佳模型权重已保存至: {target_weight_dst}")
    else:
        print("未找到训练权重文件，训练可能失败。")
        
    # 4.2 自定义可视化
    results_csv = os.path.join(train_run_dir, 'results.csv')
    if os.path.exists(results_csv):
        visualize_metrics(results_csv)
    
    print("训练流程全部结束。")

if __name__ == "__main__":
    main()