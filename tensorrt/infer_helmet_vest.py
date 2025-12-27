## 3. 推理代码 infer_helmet_vest.py
import argparse
from ultralytics import YOLO
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='待检测图片/文件夹/视频/视频文件夹')
    parser.add_argument('--save-dir', type=str, default='outputs', help='检测结果保存目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    args = parser.parse_args()

    # 加载模型
    model = YOLO(args.weights)

    # 自动创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 推理
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,           # 自动保存可视化结果
        project=str(save_dir),# 保存到指定目录
        name='results',      # 子目录名
        show=False           # 不弹窗显示
    )

    # 输出检测结果（图片/视频均支持）
    print(f'检测完成，结果已保存到: {save_dir / "results"}')
    for r in results:
        print(f'文件: {r.path}')
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            name = model.names[cls]
            print(f'  类别: {name}, 置信度: {conf:.2f}, 坐标: {xyxy}')

if __name__ == '__main__':
    main()