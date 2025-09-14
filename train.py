# pip install roboflow requests

import os
from roboflow import Roboflow
import subprocess

# ==========================
# 1. Настройки Roboflow
# ==========================
API_KEY = "Q3obyJNseh7lgkLcmWhv"
WORKSPACE = "AIcar"

# Проект 1 — Object Detection
PROJECT_OD = "car-scratch-and-dent-9ow3t"
VERSION_OD = 1

# Проект 2 — Instance Segmentation
PROJECT_SEG = "rust-and-scrach-jk5mg"
VERSION_SEG = 1

# ==========================
# 2. Инициализация Roboflow
# ==========================
rf = Roboflow(api_key=API_KEY)

# Скачать Object Detection датасет
project_od = rf.workspace(WORKSPACE).project(PROJECT_OD)
dataset_od = project_od.version(VERSION_OD).download("yolov7")
yaml_od = os.path.join(dataset_od, "data.yaml")

# Скачать Instance Segmentation датасет
project_seg = rf.workspace(WORKSPACE).project(PROJECT_SEG)
dataset_seg = project_seg.version(VERSION_SEG).download("yolov7-seg")  # для сегментации
yaml_seg = os.path.join(dataset_seg, "data.yaml")

# ==========================
# 3. Скачать YOLOv7 репозиторий (если не скачан)
# ==========================
if not os.path.exists("yolov7"):
    subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov7.git"])

# ==========================
# 4. Установить зависимости
# ==========================
subprocess.run(["python", "-m", "pip", "install", "-r", "yolov7/requirements.txt"])

# ==========================
# 5. Функция обучения
# ==========================
def train_yolov7(yaml_path, weights="yolov7/yolov7.pt", epochs=50, exp_name="exp"):
    train_cmd = [
        "python", "yolov7/train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", str(epochs),
        "--data", yaml_path,
        "--weights", weights,
        "--project", "runs/train",
        "--name", exp_name
    ]
    subprocess.run(train_cmd)

# Обучаем Object Detection
train_yolov7(yaml_od, exp_name="exp_od")

# Обучаем Instance Segmentation
train_yolov7(yaml_seg, exp_name="exp_seg")

# ==========================
# 6. Функция инференса
# ==========================
def infer_yolov7(img_path, weights_path, yaml_path):
    infer_cmd = [
        "python", "yolov7/detect.py",
        "--weights", weights_path,
        "--conf", "0.25",
        "--img-size", "640",
        "--source", img_path,
        "--data", yaml_path
    ]
    subprocess.run(infer_cmd)

# Пример инференса для Object Detection
infer_yolov7(
    img_path="путь_к_вашему_изображению.jpg",
    weights_path="runs/train/exp_od/weights/best.pt",
    yaml_path=yaml_od
)

# Пример инференса для Instance Segmentation
infer_yolov7(
    img_path="путь_к_вашему_изображению.jpg",
    weights_path="runs/train/exp_seg/weights/best.pt",
    yaml_path=yaml_seg
)
