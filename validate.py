import matplotlib

from ultralytics import YOLO

matplotlib.use("TkAgg")
if __name__ == "__main__":
    # 加载训练好的模型
    model = YOLO("runs/detect/train12/weights/best.pt")
    # 对验证集进行评估
    metrics = model.val(data="datasets/dogs/data.yaml")
