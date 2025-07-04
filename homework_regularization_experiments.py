
# 3.1 
"""
Итоговые результаты
Без регуляризации:
  Train Acc: 0.9928, Test Acc: 0.9754
  Train Loss: 0.0238, Test Loss: 0.1201
  Разрыв (train-test): 0.0174

Dropout 0.1:
  Train Acc: 0.9892, Test Acc: 0.9771
  Train Loss: 0.0340, Test Loss: 0.0994
  Разрыв (train-test): 0.0121

Dropout 0.3:
  Train Acc: 0.9820, Test Acc: 0.9801
  Train Loss: 0.0599, Test Loss: 0.0783
  Разрыв (train-test): 0.0019

Dropout 0.5:
  Train Acc: 0.9672, Test Acc: 0.9803
  Train Loss: 0.1151, Test Loss: 0.0688
  Разрыв (train-test): -0.0131

Только BatchNorm:
  Train Acc: 0.9942, Test Acc: 0.9831
  Train Loss: 0.0171, Test Loss: 0.0616
  Разрыв (train-test): 0.0111

Dropout 0.3 + BatchNorm:
  Train Acc: 0.9869, Test Acc: 0.9834
  Train Loss: 0.0389, Test Loss: 0.0538
  Разрыв (train-test): 0.0035

L2 регуляризация:
  Train Acc: 0.9813, Test Acc: 0.9752
  Train Loss: 0.0582, Test Loss: 0.0784
  Разрыв (train-test): 0.0061
"""


# 3.2 
"""
Итоговые результаты адаптивной регуляризации
Адаптивный Dropout:
  Финальная точность: 0.9832
  Разрыв train-test: 0.0093

Разный momentum BatchNorm:
  Финальная точность: 0.9835
  Разрыв train-test: 0.0108

Комбинированная регуляризация:
  Финальная точность: 0.9790
  Разрыв train-test: -0.0060

Слое-специфичная регуляризация:
  Финальная точность: 0.9820
  Разрыв train-test: 0.0112
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Config:
    input_size = 784
    num_classes = 10
    batch_size = 64
    epochs = 15
    lr = 0.001
    hidden_sizes = [512, 256]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveRegularizedModel(nn.Module):
    def __init__(self, 
                 dropout_schedule=None, 
                 batchnorm_momentum=0.1,
                 use_l2=False,
                 layer_specific_reg=False):
        super().__init__()
        
        self.dropout_schedule = dropout_schedule or [0.3, 0.3]
        self.batchnorm_momentum = batchnorm_momentum
        self.use_l2 = use_l2
        self.layer_specific_reg = layer_specific_reg
        
        self.layer1 = nn.Sequential(
            nn.Linear(Config.input_size, Config.hidden_sizes[0]),
            nn.BatchNorm1d(Config.hidden_sizes[0], momentum=self.batchnorm_momentum),
            nn.ReLU(),
            nn.Dropout(self.dropout_schedule[0])
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(Config.hidden_sizes[0], Config.hidden_sizes[1]),
            nn.BatchNorm1d(Config.hidden_sizes[1], momentum=self.batchnorm_momentum*0.5 if layer_specific_reg else self.batchnorm_momentum),
            nn.ReLU(),
            nn.Dropout(self.dropout_schedule[1] if len(self.dropout_schedule) > 1 else self.dropout_schedule[0])
        )
        
        self.classifier = nn.Linear(Config.hidden_sizes[1], Config.num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.classifier(x)
    
    def update_dropout(self, epoch):
        if len(self.dropout_schedule) == 2:
            new_p = max(0.1, self.dropout_schedule[0] * (1 - epoch/(Config.epochs*1.5)))
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = new_p

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

def train_adaptive_model(model, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, 
                         weight_decay=0.001 if model.use_l2 else 0.0)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_loss': [], 
              'train_acc': [], 'test_acc': [],
              'dropout_values': []}
    
    for epoch in range(Config.epochs):
        if hasattr(model, 'update_dropout'):
            model.update_dropout(epoch)
        
        model.train()
        train_loss, train_correct = 0.0, 0
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            data, target = data.to(Config.device), target.to(Config.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
        
        model.eval()
        test_loss, test_correct = 0.0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(Config.device), target.to(Config.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_acc = train_correct / len(train_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                history['dropout_values'].append(module.p)
                break
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    return history

experiments = [
    {'name': 'Адаптивный Dropout', 
     'model': AdaptiveRegularizedModel(dropout_schedule=[0.5, 0.3])},
    
    {'name': 'Разный momentum BatchNorm', 
     'model': AdaptiveRegularizedModel(batchnorm_momentum=0.3)},
     
    {'name': 'Комбинированная регуляризация', 
     'model': AdaptiveRegularizedModel(dropout_schedule=[0.4, 0.2], 
                                      batchnorm_momentum=0.2,
                                      use_l2=True)},
     
    {'name': 'Слое-специфичная регуляризация', 
     'model': AdaptiveRegularizedModel(dropout_schedule=[0.4, 0.2],
                                      batchnorm_momentum=0.3,
                                      layer_specific_reg=True)}
]

results = []

for exp in experiments:
    print(f"\n=== Эксперимент: {exp['name']} ===")
    model = exp['model'].to(Config.device)
    
    history = train_adaptive_model(model, train_loader, test_loader)
    results.append({
        'name': exp['name'],
        'history': history,
        'final_test_acc': history['test_acc'][-1],
        'final_gap': history['train_acc'][-1] - history['test_acc'][-1]
    })

    if 'dropout_values' in history:
        plt.figure()
        plt.plot(history['dropout_values'])
        plt.title(f'Изменение Dropout ({exp["name"]})')
        plt.xlabel('Эпоха')
        plt.ylabel('Значение Dropout')
        plt.grid(True)
        plt.savefig(f'dropout_{exp["name"].lower().replace(" ", "_")}.png')
        plt.close()

plt.figure(figsize=(12, 6))
for res in results:
    plt.plot(res['history']['test_acc'], label=res['name'])

plt.xlabel('Эпоха')
plt.ylabel('Точность на тесте')
plt.title('Сравнение адаптивных методов регуляризации')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('adaptive_regularization_comparison.png')
plt.close()

print("\n=== Итоговые результаты адаптивной регуляризации ===")
for res in results:
    print(f"{res['name']}:")
    print(f"  Финальная точность: {res['final_test_acc']:.4f}")
    print(f"  Разрыв train-test: {res['final_gap']:.4f}\n")