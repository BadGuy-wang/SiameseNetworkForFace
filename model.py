import os

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import show_plot


# 搭建模型
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),  # 镜像填充
            nn.Conv2d(1, 4, kernel_size=3),  # 卷积
            nn.ReLU(inplace=True),  # Relu激活
            nn.BatchNorm2d(4),  # 批标准化

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# 自定义ContrastiveLoss
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# 定义训练过程train
def train(net, optimizer, criterion, train_dataloader, args):
    counter = []
    loss_history = []
    iteration_number = 0
    # 开始训练
    for epoch in range(1, args.epochs+1):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            # img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()  # 数据移至GPU
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))
    save_model(net, 'siameseNet' + str(epoch), 'checkpoint/')
    show_plot(counter, loss_history)


def save_model(model, model_name, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), save_dir + model_name + '.pth')
    else:
        print('Model is error')
