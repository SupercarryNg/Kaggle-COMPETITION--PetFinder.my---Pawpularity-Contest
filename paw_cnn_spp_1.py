from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from paw_model import *
from get_data import *


# 设置文件路径，获得数据
rootdir = 'petfinder-pawpularity-score'
subdir_train = 'train'
train_dataset = MyData(rootdir, subdir_train)

# 装载数据
train = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

# 打印数据集size
print('训练集大小为{}'.format(len(train)))

# 导入模型
model = PawModel()
model = model.cuda()

# 定义各种初始化参数
learning_rate = 1e-2
epochs = 30
error = nn.MSELoss()
error = error.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_step = 0

# 添加tensorboard
writer = SummaryWriter('logs')

# 训练模型
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

        writer.add_scalar('train_loss', loss.item(), train_step)
        if train_step % 100 == 0:
            print('第{}次迭代的loss为: {}'.format(train_step, loss.item()))

    # model.eval()
    # total_test_loss = 0
    # with torch.no_grad():
    #     for data in test:
    #         images, labels = data
    #         images = img_transform(images)
    #         images = images.cuda()
    #         labels = labels.cuda()
    #         output = model(images)
    #
    #         # 计算损失函数，并更新权重
    #         loss = error(output, labels)
    #         total_test_loss = total_test_loss + loss.item()
    # print('---------------------------------')
    # print('第{}轮训练的测试集loss为：{}'.format(epoch+1, total_test_loss))
    # print('---------------------------------')
    # writer.add_scalar('test_loss', total_test_loss, epoch)

writer.close()