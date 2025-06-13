import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
# 从指定路径加载 CIFAR10 训练集和测试集
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

# 输出数据集长度信息
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集，训练集开启随机打乱
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 定义改进的 AlexNet 模型
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # 调整第一个卷积层的步长和输出通道数
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32x16x16
            # 调整第二个卷积层的输出通道数
            nn.Conv2d(32, 128, kernel_size=3, padding=1),  # 128x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 128x8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 256x4x4
        )
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 调整全连接层的输入维度
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 设备配置，优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModifiedAlexNet().to(device)
print("使用设备:", device)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 训练参数设置
total_train_step = 0
total_test_step = 0
epoch = 10

# 使用 TensorBoard 记录训练过程
writer = SummaryWriter("logs_train")

start_time = time.time()

# 开始训练循环
for i in range(epoch):
    print(f"-----第{i + 1}轮训练开始-----")
    model.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
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

    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
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

    torch.save(model.state_dict(), f"model_save/modified_alexnet_{i + 1}.pth")
    print(f"模型已保存: modified_alexnet_{i + 1}.pth")

writer.close()
print("训练完成!")