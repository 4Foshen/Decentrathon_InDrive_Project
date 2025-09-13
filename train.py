# pip install roboflow requests

import os
from roboflow import Roboflow
import requests
import subprocess

# 1. Импортировать датасет через Roboflow API
API_KEY = "Q3obyJNseh7lgkLcmWhv"
WORKSPACE = "ВАШ_WORKSPACE"
PROJECT = "ВАШ_PROJECT"
VERSION = 1

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset_path = version.download("yolov7")  # скачает и распакует архив

# 2. Найти data.yaml
yaml_path = os.path.join(dataset_path, "data.yaml")

# 3. Скачать YOLOv7 репозиторий, если не скачан
if not os.path.exists("yolov7"):
    subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov7.git"])

# 4. Установить зависимости
subprocess.run(["python", "-m", "pip", "install", "-r", "yolov7/requirements.txt"])

# 5. Запустить обучение
train_cmd = [
    "python", "yolov7/train.py",
    "--img", "640",
    "--batch", "16",
    "--epochs", "50",
    "--data", yaml_path,
    "--weights", "yolov7/yolov7.pt"
]
subprocess.run(train_cmd)

# 6. Инференс: загрузить изображение и предсказать класс
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

# Пример инференса (замените путь к изображению и весам на свои)
infer_yolov7(
    img_path="путь_к_вашему_изображению.jpg",
    weights_path="runs/train/exp/weights/best.pt",
    yaml_path=yaml_path
)
