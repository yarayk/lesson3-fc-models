# 1.1

"""
Графики были проанализированы

Время тренировки: 69.69 seconds
Финальная точность тренировки: 0.9797
Финальная точность тестирования: 0.9746
""" 

# 1.2

"""
Без регуляризации:
Уже на 2-3 слоях сеть достигает почти максимальной точности, дальнейшее увеличение глубины даёт маленький прирост.
Большой разрыв между точностью на train и test. 99.4% и 97.8% для 2 слоёв.
c регуляризацией (Dropout + BatchNorm):
Разрыв между train и test становится меньше. 98.1% и 97.8% для 2 слоёв.
Лучший результат: 3 слоя c Dropout+BatchNorm 98.2% на тесте.
Вывод: быстро и эффективно, хватит 2-3 слоёв c регуляризацией.
Без регуляризации переобучение начинается уже c 2 слоёв, c регуляризацией глубже 3 слоёв точность почти не растёт.
"""


import time
import torch
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_config(depth, use_reg=False):
    layers = []
    input_size = 784
    hidden_size = 128
    for i in range(depth - 1):
        layers.append({"type": "linear", "size": hidden_size})
        if use_reg:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if use_reg:
            layers.append({"type": "dropout", "rate": 0.2})
    return {
        "input_size": input_size,
        "num_classes": 10,
        "layers": layers
    }

depths = [1, 2, 3, 5, 7]
configs = {
    "No Reg": False,
    "Dropout+BatchNorm": True
}

train_loader, test_loader = get_mnist_loaders(batch_size=64)

for reg_name, use_reg in configs.items():
    print(f"\n=== Регуляризация: {reg_name} ===")
    for d in depths:
        print(f"\nМодель глубиной {d} слоёв")
        config = create_config(d, use_reg)
        model = FullyConnectedModel(
            input_size=config["input_size"],
            num_classes=config["num_classes"],
            layers=config["layers"]
        ).to(device)

        print(f"Параметров: {count_parameters(model)}")
        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))
        end = time.time()

        print(f"Время обучения: {end - start:.2f} сек")
        print(f"Точность на train: {history['train_accs'][-1]:.4f}")
        print(f"Точность на test: {history['test_accs'][-1]:.4f}")

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title(f"Train Accuracy (Depth {d}, {reg_name})")
        plt.plot(history['train_accs'], label='Train Acc')
        plt.plot(history['test_accs'], label='Test Acc')
        plt.legend()

        plt.subplot(1,2,2)
        plt.title(f"Loss (Depth {d}, {reg_name})")
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['test_losses'], label='Test Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()