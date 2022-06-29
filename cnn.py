# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import *
import matplotlib.pyplot as plt
import numpy as np
# %%

# 数据集的预处理
from torchvision.datasets import mnist

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # x是一个batch_size的数据
        # x:1*28*28
        x = F.relu(self.conv1(x))
        # 20*24*24
        x = F.max_pool2d(x, 2, 2)
        # 20*12*12
        x = F.relu(self.conv2(x))
        # 50*8*8
        x = F.max_pool2d(x, 2, 2)
        # 50*4*4
        x = x.view(-1, 50 * 4 * 4)
        # 压扁成了行向量，(1,50*4*4)
        x = F.relu(self.fc1(x))
        # (1,500)
        x = self.fc2(x)
        # (1,10)
        return F.log_softmax(x, dim=1)

def load_mnist(path):
    data_path = path
    data_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )

    # 获取数据集
    train_data = mnist.MNIST(data_path, train=True, transform=data_tf, download=True)
    test_data = mnist.MNIST(data_path, train=False, transform=data_tf, download=True)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64)

    return train_loader, test_loader

def train(model, device, train_loader, optimizer, epoch):
    losses = []
    begin_time = time()
    model.train()
    for idx, (t_data, t_target) in enumerate(train_loader):
        t_data, t_target = t_data.to(device), t_target.to(device)
        pred = model(t_data)  # batch_size*10
        loss = F.nll_loss(pred, t_target)

        optimizer.zero_grad()  # 将上一步的梯度清0
        loss.backward()  # 重新计算梯度
        optimizer.step()  # 更新参数
        if idx % 50 == 0:
            print("epoch:{},loss:{}".format(epoch, loss.item()))
            losses.append(loss.item())  # 每100批数据采样一次loss，记录下来，用来画图可视化分析。
    end_time = time()
    print(f"模型训练时间+测试时间：{end_time - begin_time}")
    torch.save(model, './CNN2_model.pth')

    len_l = len(losses)
    x = [i for i in range(len_l)]
    figure = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x, losses)
    plt.xlabel('n')
    plt.ylabel('Loss')
    plt.show()

def cnn_predict(testImgs, testLabels):
    #加载模型
    model = torch.load('CNN_model.pth')
    model.eval()
    #改变数据类型
    #testImgs = testImgs.reshape(-1, 28, 28)
    inputs = torch.from_numpy(testImgs).to(torch.float32)

    pred = model(inputs.reshape(-1, 1, 28, 28))
    pred_class = pred.argmax(dim=1)
    # 打印模型正确率
    print("accuracy: ", accuracy_score(pred_class.numpy(), np.array(testLabels)))
    return pred_class.numpy()

# 训练相关参数设置
# lr = 0.01
# momentum = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
# model = CNN().to(device)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# data_path = './mnist'
# train_data, test_data = load_mnist(data_path)
# batch_size = 64
#
#
# num_epochs = 3
# # 记录起来用来画图的，可以画出损失随着迭代次数而下降。
#
# # 测试我们的模型训练要花多久。
# train_loader, test_loader = load_mnist(data_path)
# for epoch in range(num_epochs):
#     train(model, device, train_loader, optimizer, epoch)





