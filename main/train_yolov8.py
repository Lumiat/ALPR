import os
import shutil
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO

# ================= 配置区域 =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据集路径
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ccpd_yolo")
# 结果保存路径
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
# 权重保存路径
WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weight")

MODEL_NAME = 'yolov8n-pose.pt' 

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO Pose训练脚本 - 车牌四点检测')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像尺寸 (默认: 640)')
    
    # 可选参数
    parser.add_argument('--device', type=int, default=0,
                        help='GPU设备编号 (默认: 0, 使用cpu则设为-1)')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt',
                        help='预训练模型名称 (默认: yolov8n-pose.pt)')
    
    return parser.parse_args()

def ensure_dirs():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

def create_yaml():
    yaml_path = os.path.join(PROJECT_ROOT, "dataset", "ccpd_plate.yaml")
    data_config = {
        'path': DATASET_DIR,       
        'train': 'images/train',   
        'val': 'images/val',       
        'names': {0: 'license_plate'}, 
        'kpt_shape': [4, 3],       
        'flip_idx': [1, 0, 3, 2]   
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
    print(f"数据配置文件已生成: {yaml_path}")
    return yaml_path

def visualize_metrics(csv_path):
    """
    读取训练日志并可视化保存 (已修正：增加 Pose 指标)
    """
    print("正在绘制训练指标可视化图表...")
    try:
        # 读取 YOLO 输出的 results.csv
        df = pd.read_csv(csv_path)
        # 去除列名两端的空格 (YOLO输出的csv表头通常带有空格)
        df.columns = [c.strip() for c in df.columns]
        
        # === 修正部分：添加了 Pose 相关的 mAP 指标 ===
        metrics_map = {
            # 损失函数
            'train/box_loss': 'Box Loss (Training)',
            'train/pose_loss': 'Keypoint Loss (Training)', # 关键！这是 Pose Loss
            
            # Box 检测指标 (衡量能不能框住车牌)
            'metrics/mAP50(B)': 'Box mAP@0.5',
            'metrics/mAP50-95(B)': 'Box mAP@0.5:0.95',
            
            # Pose 关键点指标 (衡量角点定得准不准，即矫正准不准)
            'metrics/mAP50(P)': 'Pose mAP@0.5',         # 新增
            'metrics/mAP50-95(P)': 'Pose mAP@0.5:0.95'  # 新增
        }
        
        for col, title in metrics_map.items():
            if col in df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(df['epoch'], df[col], marker='o', markersize=2, linestyle='-')
                plt.title(f'Metric: {title}')
                plt.xlabel('Epochs')
                plt.ylabel(title)
                plt.grid(True)
                
                # 保存文件名格式: {metric}_visualize.png
                safe_name = col.replace('/', '_').replace('(', '').replace(')', '')
                save_path = os.path.join(RESULT_DIR, f"{safe_name}_visualize.png")
                plt.savefig(save_path)
                plt.close()
                print(f"已保存: {save_path}")
            else:
                # 只有当确实找不到该指标时才警告
                print(f"提示: 日志中未找到指标 {col} (可能训练尚未生成该列)")
                
    except Exception as e:
        print(f"可视化失败: {e}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 打印训练配置
    print("=" * 50)
    print("训练配置:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Device: {args.device}")
    print(f"  Model: {args.model}")
    print("=" * 50)
    
    ensure_dirs()
    
    # 1. 创建配置文件
    yaml_path = create_yaml()
    
    # 2. 加载模型
    print(f"加载模型: {args.model}...")
    model = YOLO(args.model) 
    
    # 3. 开始训练
    print("开始训练...")
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        project=os.path.join(PROJECT_ROOT, 'runs'),
        name='ccpd_pose_train',
        exist_ok=True,   
        plots=True,      
        device=args.device         
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
