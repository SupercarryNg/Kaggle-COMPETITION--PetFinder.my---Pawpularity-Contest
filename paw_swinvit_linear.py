import timm
import torch
import torch.nn as nn
from get_data_linear import *
from torch.utils.data import DataLoader, random_split


class Swinvitlinear(nn.Module):
    def __init__(self):
        super(Swinvitlinear, self).__init__()
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=0)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        outputs = self.fc2(x)
        return outputs


model = Swinvitlinear()
model = model.cuda()
# 设置文件路径，获得数据
rootdir = 'petfinder-pawpularity-score'
subdir_train = 'train'
dataset = MyData(rootdir, subdir_train)
train_dataset, valid_dataset = random_split(dataset, [9000, 912], generator=torch.Generator().manual_seed(42))
print('length of trainset is {}'.format(len(train_dataset)))
print('length of validset is {}'.format(len(valid_dataset)))

# 装载数据
train = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
valid = valid_dataset

# 打印数据集size
print('训练集大小为{}'.format(len(train)))

# 定义各种初始化参数
batch_size = 16
learning_rate = 1e-5
epochs = 30
error = nn.BCEWithLogitsLoss()
error = error.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_step = 0


# 定义sigmoid函数，需要将output走一层sigmoid才能作为预测结果
def sigmoid(x):
    sigmoid_x = 1/(1+np.exp(-x))
    return sigmoid_x


for epoch in range(epochs):
    print('--------第 {} 轮训练开始--------'.format(epoch+1))

    model.train()
    for data in train:
        # 将训练集中的images和label分别传入
        images, labels = data
        labels = labels.view(-1, 1)  # 给label加多一个维度，统一output和label的维度
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
            print('第{}次迭代的loss为: {}'.format(train_step, loss.item()))

    model.eval()
    with torch.no_grad:
        res = []
        for valid_data in valid:
            img, label = valid_data
            img = torch.reshape(img, (1, 3, 224, 224))
            img = img.cuda()
            pred = model(img)
            pred = sigmoid(pred)
            res.append((pred - label) ** 2)
            if len(res) % 100 == 0:
                print('进行到第{}张图片的验证'.format(len(res)))
                print('pred = {}'.format(pred), end='//')
                print('label = {}'.format(label))
                print(res[-1])
        total_RMSE = np.sqrt(sum(res) / len(res))
        print('The Total RMSE for {}th epoch is {}'.format(epoch, total_RMSE))

