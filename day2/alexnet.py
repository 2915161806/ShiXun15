import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root="D:\\PycharmProjects\\day1\\day02\\dataset_chen",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="D:\\PycharmProjects\\day1\\day02\\dataset_chen",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64x8x8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 192x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 192x4x4
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 384x4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 256x4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x4x4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 256x2x2
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chen = AlexNet().to(device)
print("使用设备:", device)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate, momentum=0.9)

# 训练参数
total_train_step = 0
total_test_step = 0
epoch = 10

# Tensorboard
writer = SummaryWriter("logs_train")

start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    chen.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}步的训练loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(f"本轮训练时间: {end_time - start_time:.2f}秒")

    chen.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = chen(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_accuracy / test_data_size

    print(f"测试集平均loss: {avg_test_loss:.4f}")
    print(f"测试集正确率: {test_accuracy:.4f}")

    writer.add_scalar("test_loss", avg_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    torch.save(chen.state_dict(), f"model_save/alexnet_{i+1}.pth")
    print(f"模型已保存: alexnet_{i+1}.pth")

writer.close()
print("训练完成!")
