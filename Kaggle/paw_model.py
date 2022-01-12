import torch.nn as nn
import torch.nn.functional as F
import torch


class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()  # bs表示batch_size, c表示channel, h表示高度, w表示宽度
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)  # dim=-1 表示按照最后一个维度来进行拼接
        return x


class PawModel(nn.Module):
    def __init__(self, spp_level=5):
        super(PawModel, self).__init__()
        self.spp_level = spp_level

        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2 ** (i * 2)
        print(self.num_grids)
        # 卷积层 池化层
        self.conv_model = nn.Sequential(

            nn.Conv2d(3, 32, (5, 5), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5, 5), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 5), 1, 2),
            nn.ReLU()

        )
        # 池化金字塔层 >>> 统一输入全连接层的input dim
        self.spp_layer = SPPLayer(spp_level)

        # 全连接层
        self.linear_model = nn.Sequential(
            nn.Linear(self.num_grids*64, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv_model(x)
        x = self.spp_layer(x)
        output = self.linear_model(x)
        return output


if __name__ == '__main__':
    jason = PawModel()
    inputtest = torch.ones((64, 3, 720, 720))
    outputtest = jason(inputtest)
    print(outputtest.shape)
