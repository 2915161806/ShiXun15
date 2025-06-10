import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ------------ 数据准备部分 ------------
# 加载 CIFAR10 训练集
train_data = torchvision.datasets.CIFAR10(
    root="D:\\PycharmProjects\\day1\\day02\\dataset_chen",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
# 加载 CIFAR10 测试集
test_data = torchvision.datasets.CIFAR10(
    root="D:\\PycharmProjects\\day1\\day02\\dataset_chen",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 输出训练集和测试集的长度
train_size = len(train_data)
test_size = len(test_data)
print(f"训练集长度: {train_size}")
print(f"测试集长度: {test_size}")

# 创建数据加载器，训练集随机打乱
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# ------------ 模型定义部分 ------------
# 定义一个新的 AlexNet 变体模型
class NewAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取模块
        self.feature_extractor = nn.Sequential(
            # 第一个卷积层，使用较小的输出通道数和步长
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 16x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 16x16x16
            # 第二个卷积层，输出通道数有所调整
            nn.Conv2d(16, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64x8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 128x4x4
        )
        # 分类模块
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 全连接层输入维度根据特征提取结果调整
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ------------ 设备和训练配置部分 ------------
# 选择设备，优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NewAlexNet().to(device)
print(f"当前使用设备: {device}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# 训练参数初始化
train_steps = 0
test_steps = 0
epochs = 10

# 使用 TensorBoard 记录训练信息
tb_writer = SummaryWriter("logs_train")

start = time.time()

# ------------ 训练和测试循环部分 ------------
for epoch in range(epochs):
    print(f"--- 第 {epoch + 1} 轮训练开始 ---")
    net.train()
    for batch_data in train_loader:
        images, labels = batch_data
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_steps += 1
        if train_steps % 500 == 0:
            print(f"第 {train_steps} 步训练损失: {loss.item():.4f}")
            tb_writer.add_scalar("train_loss", loss.item(), train_steps)

    end = time.time()
    print(f"本轮训练耗时: {end - start:.2f} 秒")

    net.eval()
    total_test_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch_data in test_loader:
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            correct = (outputs.argmax(1) == labels).sum()
            total_correct += correct.item()

    avg_loss = total_test_loss / len(test_loader)
    accuracy = total_correct / test_size

    print(f"测试集平均损失: {avg_loss:.4f}")
    print(f"测试集准确率: {accuracy:.4f}")

    tb_writer.add_scalar("test_loss", avg_loss, test_steps)
    tb_writer.add_scalar("test_accuracy", accuracy, test_steps)
    test_steps += 1

    torch.save(net.state_dict(), f"model_save/new_alexnet_{epoch + 1}.pth")
    print(f"模型已保存: new_alexnet_{epoch + 1}.pth")

tb_writer.close()
print("训练结束!")