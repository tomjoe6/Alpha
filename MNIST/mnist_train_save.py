import mnist_function
import torch
import torch.nn as nn
import torch.optim as op
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image,ImageOps
import os
from typing import cast

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    # 训练和推理必须使用同一套归一化策略, 否则分布不一致会掉点
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./mnistdata', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(3136, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # 64*7*7 = 3136
        x = x.view(-1, 3136)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    # 切换到训练模式: Dropout 生效
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # 反向传播计算梯度
        loss.backward()
        # 参数更新
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    print(f'Epoch{epoch}/{EPOCHS}-Train Loss:{train_loss:.4f},Train ACC:{train_acc:.4f}%')


def test(model, device, test_loader, criterion):
    # 切到评估模式: Dropout 关闭
    model.eval()
    test_loss = 0.0
    correct = 0

    # 测试阶段不需要梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f'Test Loss:{test_loss:.4f},Test Accuract:{correct}/{len(test_loader.dataset)}({accuracy:.4f}%)\n')
    return accuracy

if __name__ == '__main__':
    print(f'Using device:{DEVICE}')

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = op.Adam(model.parameters(), lr=LEARNING_RATE)

    final_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, criterion, epoch)
        final_accuracy = test(model, DEVICE, test_loader, criterion)
        
    # 只保存权重(推荐), 加载时配合 load_state_dict
    path='weights/mnist_cnn_epoch1_1.pth'

    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')
    print(f'Final Acc:{final_accuracy:.4f}%')

#仅加载权重不训练代码       
# 实例化模型
#model = CNN().to(DEVICE)
# 加载权重
#model.load_stat·   e_dict(torch.load('mnist_cnn.pth'))
# 直接预测
#predict_local_image("图片路径", model, DEVICE)