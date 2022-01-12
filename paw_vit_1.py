import timm
import torch
import torch.nn as nn
from get_data import *

from torch.utils.data import DataLoader

model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=16)
model = model.cuda()
# 设置文件路径，获得数据
rootdir = 'petfinder-pawpularity-score'
subdir_train = 'train'
train_dataset = MyData(rootdir, subdir_train)

# 装载数据
train = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

# 打印数据集size
print('训练集大小为{}'.format(len(train)))

# 定义各种初始化参数
batch_size = 2
learning_rate = 1e-5
epochs = 30
error = nn.CrossEntropyLoss()
error = error.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_step = 0

for epoch in range(epochs):
    print('--------第 {} 轮训练开始--------'.format(epoch+1))

    model.train()
    for data in train:
        # 将训练集中的images和label分别传入
        images, labels = data
        # labels = labels.view(-1, 1)  # 给label加多一个维度，统一output和label的维度
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)

        # 计算损失函数，并更新权重
        loss = error(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step = train_step+1

        if train_step % 100 == 0:
            print('第{}次迭代的loss为: {}'.format(train_step, loss.item()), end=r' // ')
            accuracy = (output.argmax(1) == labels).sum()
            print('第{}次迭代的accuracy为: {}'.format(train_step, accuracy.item()/batch_size))


