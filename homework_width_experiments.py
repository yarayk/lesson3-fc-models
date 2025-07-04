# 2.1 

"""
Точность:
Средние сети показали лучший баланс 97.1% на тесте. Более широкие модели дали очень маленький прирост.
Параметры:
Количество параметров растет экспоненциально.
Время обучения:
Все модели обучались 55-65 секунд. Широкие сети немного медленнее узких.

Вывод: Для MNIST достаточно средней ширины
"""

# 2.2

"""
Лучшая архитектура:
Конфигурация: 512-512-512
Точность: 0.9693
Параметры: 932,362
Схема: Постоянная
"""


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, num_classes, layers):
        super().__init__()
        self.layers = nn.Sequential(*self._build_layers(input_size, layers))
        self.classifier = nn.Linear(layers[-2]['size'], num_classes)
        
    def _build_layers(self, input_size, layers):
        module_list = []
        prev_size = input_size
        for layer in layers:
            if layer['type'] == 'linear':
                module_list.append(nn.Linear(prev_size, layer['size']))
                prev_size = layer['size']
            elif layer['type'] == 'relu':
                module_list.append(nn.ReLU())
        return module_list
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return self.classifier(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

width_options = [64, 128, 256, 512]
scheme_names = ['Сужение', 'Расширение', 'Постоянная']
results = []

for w1, w2, w3 in tqdm(list(product(width_options, repeat=3)), desc='Перебор архитектур'):
    if w1 > w2 > w3:
        scheme = 'Сужение'
    elif w1 < w2 < w3:
        scheme = 'Расширение'
    elif w1 == w2 == w3:
        scheme = 'Постоянная'
    else:
        continue
        
    model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": w1},
            {"type": "relu"},
            {"type": "linear", "size": w2},
            {"type": "relu"}, 
            {"type": "linear", "size": w3},
            {"type": "relu"}
        ]
    ).to(device)
    
    params = count_parameters(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    
    results.append({
        'Архитектура': f"{w1}-{w2}-{w3}",
        'Параметры': params,
        'Точность': accuracy,
        'Схема': scheme
    })

df = pd.DataFrame(results)
best_model = df.loc[df['Точность'].idxmax()]

print("\nЛучшая архитектура:")
print(f"Конфигурация: {best_model['Архитектура']}")
print(f"Точность: {best_model['Точность']:.4f}")
print(f"Параметры: {best_model['Параметры']:,}")
print(f"Схема: {best_model['Схема']}")

plt.figure(figsize=(12, 6))
for scheme in scheme_names:
    subset = df[df['Схема'] == scheme]
    plt.scatter(subset['Параметры'], subset['Точность'], label=scheme, s=100)

plt.xlabel('Количество параметров')
plt.ylabel('Точность на тесте')
plt.title('Сравнение архитектур нейронной сети')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('architecture_results.png')
plt.close()

df.to_csv('architecture_results.csv', index=False)