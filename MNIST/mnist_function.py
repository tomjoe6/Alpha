import torch
import torch.nn as nn
import torch.optim as op
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter, ImageOps, ImageStat
import os
from typing import cast

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
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
        # Conv+Pool 两次后形状: 1x28x28 -> 32x14x14 -> 64x7x7
        x = self.pool(self.relu(self.conv(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # 64*7*7 = 3136, 必须和 fc1 的输入维度一致
        x = x.view(-1, 3136)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        # 返回 logits (未做 softmax), 便于配合 CrossEntropyLoss
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    # train() 会启用 Dropout 等训练行为
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # 反向传播会把当前 batch 的梯度写入每个参数的 .grad
        loss.backward()
        # 根据 .grad 更新参数
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    print(f'Epoch{epoch}/{EPOCHS}-Train Loss:{train_loss:.4f},Train ACC:{train_acc:.4f}%')


def test(model, device, test_loader, criterion):
    # eval() 关闭 Dropout 等随机性, 保证评估稳定
    model.eval()
    test_loss = 0.0
    correct = 0

    # no_grad() 关闭 autograd, 测试时更省显存和时间
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 这里是按 batch 求平均 loss
    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f'Test Loss:{test_loss:.4f},Test Accuract:{correct}/{len(test_loader.dataset)}({accuracy:.4f}%)\n')
    return accuracy

def predict_local_image(image_path, model, device):
    """识别单张本地图片"""
    if not os.path.exists(image_path):
        print(f"错误: 找不到图片 {image_path}")
        return
    
    # 1. 打开图片并统一为灰度图
    with Image.open(image_path) as img:
        img = img.convert('L')
    img = ImageOps.autocontrast(img)

    # 2. 自动判断背景，统一成 MNIST 的黑底白字
    # 均值高通常表示白底黑字, 反相后更接近 MNIST(黑底白字)
    if ImageStat.Stat(img).mean[0] > 127:
        img = ImageOps.invert(img)

    # 3. 先缩放到 28x28，再做二值化，减少背景噪声
    img = ImageOps.pad(img, (28, 28), method=Image.Resampling.BILINEAR, color=0)

    # 4. 如果笔画太细，自动做一次或两次膨胀（MaxFilter）
    # 前景占比很小通常意味着笔画太细, 需要轻度膨胀
    foreground_ratio = sum(px > 0 for px in img.getdata()) / (28 * 28)
    if foreground_ratio < 0.08:
        img = img.filter(ImageFilter.MaxFilter(3))
    if foreground_ratio < 0.04:
        img = img.filter(ImageFilter.MaxFilter(3))

    # 5. 预处理（和训练归一化保持一致）
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    img_tensor = cast(torch.Tensor, to_tensor(img))
    # 先做阈值二值化再标准化, 可以压制背景噪声
    img_tensor = (img_tensor >= (48.0 / 255.0)).float()
    img_tensor = cast(torch.Tensor, normalize(img_tensor))
    
    # 6. 增加Batch维度
    # PyTorch模型期待的输入形状是 (batch_size, channels, height, width)
    # 当前img_tensor的形状是 (1, 28, 28)，我们需要在前面加一个维度变成 (1, 1, 28, 28)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 7. 模型推理
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        
        # softmax 把 logits 转为概率分布(各类概率和为1)
        probabilities = torch.softmax(output, dim=1)
        # 获取预测类别和置信度
        confidence, predicted = probabilities.max(1)
        
    print(f"图片: {image_path} -> 预测数字: {predicted.item()}, 置信度: {confidence.item()*100:.4f}%")
    return predicted.item()

#可以尝试修改成命令行读入图片位置，更灵活