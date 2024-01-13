# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
%matplotlib inline

# 读取训练数据集、测试数据集和样本提交文件
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
sample = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

# 划分训练集和验证集
X = train.drop('label', axis=1).to_numpy(dtype='float32')
y = train['label'].values.astype('int64')
train_X, validation_X, train_y, validation_y = train_test_split(X, y, test_size=0.15, random_state=32)

# 数据形状检查
assert X.shape[0] == y.shape[0], 'X and y must have the same number of rows'

# 转换测试集的数据类型和形状
test = test.to_numpy(dtype='float32')

# 可视化训练集中的前10个数字
fig, ax = plt.subplots(1, 10, figsize=(12, 8))
for i in range(10):
    digit = train_X[i].reshape(28, 28)
    ax[i].imshow(digit, cmap=plt.cm.binary, interpolation='bilinear')
    ax[i].set_axis_off()
    ax[i].set_title(train_y[i])
plt.show()

# 获取类别数量
no_class = len(set(y))

# 数据转换和预处理
X_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.unsqueeze(0))
])

# 定义自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image).float()
        image = image.view(1, 28, 28)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
            return image, label
        return image

# 创建训练集、验证集和预测集的数据加载器
train_set = ImageDataset(train_X, train_y, X_transform)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

validation_set = ImageDataset(validation_X, validation_y, X_transform)
validation_dataloader = DataLoader(validation_set, batch_size=32, shuffle=True)

prediction_set = ImageDataset(test, None, X_transform)
prediction_dataloader = DataLoader(prediction_set, batch_size=32)

# 获取训练集中的图像和标签
train_images, train_labels = next(iter(train_dataloader))
train_images.shape

# 获取预测集中的图像
predicted_images = next(iter(prediction_dataloader))
predicted_images.shape

# 定义手写数字识别的神经网络模型
class DigitRecognizerNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2, padding=1)

        # Dropout 层
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.6)

        # 全连接层
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        x = self.pool(F.silu(self.bn2(self.conv2(x))))
        x = self.pool(F.silu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)

        x = F.silu(self.fc1(x))
        x = self.dropout2(x)
        x = F.silu(self.fc2(x))
        x = self.dropout3(x)

        x = F.softmax(self.fc3(x))
        return x

# 实例化模型并将其移到设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitRecognizerNN().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

# 训练循环
epochs = 10

# 遍历每个 epoch
for epoch in range(epochs):
    # 打印当前 epoch 编号
    print(f"Epoch {epoch + 1}")

    # 执行当前 epoch 的训练过程
    train(train_dataloader, model, loss_fn, optimizer, device)

    # 在完成一个 epoch 后使用调度器更新学习率
    scheduler.step()

# 训练后进行验证
validate(validation_dataloader, model, loss_fn, device)

# 所有 epoch 完成后，打印结束训练的消息
print("End of training. Saving model...")

# 将训练好的模型状态字典保存到名为 "model.pth" 的文件中
torch.save(model.state_dict(), "model.pth")

# 再次进行验证
validate(validation_dataloader, model, loss_fn, device)

# 将模型设置为评估模式
model.eval()

# 用于存储预测标签的列表
predictions = []
true_labels = []

# 由于只进行预测，因此禁用梯度计算
with torch.no_grad():
    # 遍历测试数据加载器
    for data in prediction_dataloader:
        # 将输入数据移动到与模型相同的设备上
        data = data.to(device)

        # 从模型获取预测
        outputs = model(data)
        # 通过找到最大值的索引来提取预测标签
        _, predicted = torch.max(outputs.data, 1)
        # 将预测标签扩展到列表中（如果需要，移动到 CPU 上）
        predictions.extend(predicted.cpu().tolist())

        # 扩展真实标签列表（如果需要，移动到 CPU 上）
        true_labels.extend(data.cpu().tolist())

# 创建一个包含 ImageId 和预测标签的提交 DataFrame
submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),  # ImageId 从 1 开始
    "Label": predictions  # 预测的标签
})

# 将提交 DataFrame 保存到名为 'digits_nn.csv' 的 CSV 文件中，不包含索引
submission.to_csv('digits_nn.csv', index=False)

# 从数据集中获取 20 个随机索引的随机选择
random_indices = random.sample(range(len(prediction_dataloader.dataset)), 20)

# 定义子图网格的行数和列数
num_rows = 10
num_cols = 10

# 创建一个图形并设置其大小
plt.figure(figsize=(20, 20))

# 遍历图像的索引（用你的数据替换这一部分）
for idx in range(num_rows * num_cols):
    plt.subplot(num_rows, num_cols, idx + 1)
    plt.imshow(prediction_dataloader.dataset[idx][0].squeeze(), cmap='gray')  # 显示图像
    plt.axis('off')  # 隐藏坐标轴
    plt.title(f"Predicted: {predictions[idx]}")  # 将预测标签显示为标题

plt.tight_layout()  # 调整布局
plt.show()  # 显示图形