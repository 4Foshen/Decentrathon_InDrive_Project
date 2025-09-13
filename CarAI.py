import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# Пример простой нейросети для классификации состояния автомобиля
class CarStateNet(nn.Module):
	def __init__(self):
		super(CarStateNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.AdaptiveAvgPool2d((4, 4))  # фиксируем выходной размер
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 4 * 4, 128),
			nn.ReLU(),
			nn.Linear(128, 4)  # 4 класса: чистый-целый, чистый-битый, грязный-целый, грязный-битый
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x

def predict_car_state(image_path, model, device):
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.CenterCrop(128),
		transforms.ToTensor()
	])
	image = Image.open(image_path).convert('RGB')
	image = transform(image).unsqueeze(0).to(device)
	with torch.no_grad():
		outputs = model(image)
		_, predicted = torch.max(outputs, 1)
	# Классы: 0 - чистый/целый, 1 - чистый/битый, 2 - грязный/целый, 3 - грязный/битый
	classes = [
		('Чистый', 'Целый'),
		('Чистый', 'Битый'),
		('Грязный', 'Целый'),
		('Грязный', 'Битый')
	]
	return classes[predicted.item()]

if __name__ == "__main__":
	# Открыть диалоговое окно для выбора изображения
	root = tk.Tk()
	root.withdraw()
	image_path = filedialog.askopenfilename(
		title="Выберите изображение автомобиля",
		filetypes=[("Все файлы", "*.*")]
	)
	if not image_path:
		print("Файл не выбран.")
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = CarStateNet().to(device)
		model.eval()
		# Для теста используются случайные веса, обучите модель на своём датасете!
		state = predict_car_state(image_path, model, device)
		print(f"Чистота: {state[0]}")
		print(f"Целостность: {state[1]}")
