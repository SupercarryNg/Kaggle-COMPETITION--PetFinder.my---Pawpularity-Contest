import os.path

import numpy as np
import timm
import torch
from get_data_linear import *
from PIL import Image
from torch import nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# 原模型
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


model_test = torch.load('model_trained/model_linear_17.pth')
print(model_test)

# 载入transform的参数
model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
config = resolve_data_config({}, model=model)
imgTransform = create_transform(**config)


# 定义sigmoid函数
def sigmoid(x):
    sigmoid_x = 1/(1+np.exp(-x))
    return sigmoid_x


# # 模型验证
# model_test.eval()
# with torch.no_grad():
#     img_path = 'petfinder-pawpularity-score/train/0b4a913c0eb9e8e828db599196258a45.jpg'
#     img = Image.open(img_path)
#     image = imgTransform(img)
#     image = torch.reshape(image, (1, 3, 224, 224))
#     image = image.cuda()
#     output = model_test(image).item()
#     output = sigmoid(output)
#     output = output * 99 + 1
#     print(output)

def modeltest(image):
    output = model_test(image).item()
    output = sigmoid(output)
    output = output * 99 + 1
    return output


if __name__ == '__main__':
    rootdir = 'petfinder-pawpularity-score'
    subdir = 'train'
    traindataset = MyData(rootdir, subdir)
    res = []
    for data in traindataset:
        img, label = data
        img = torch.reshape(img, (1, 3, 224, 224))
        img = img.cuda()
        label = label * 99 + 1
        pred = modeltest(img)
        res.append((pred - label) ** 2)
        if len(res) % 100 == 0:
            print('进行到第{}张图片的验证'.format(len(res)))
            print('pred = {}'.format(pred), end='//')
            print('label = {}'.format(label))
            print(res[-1])
    total_RMSE = np.sqrt(sum(res) / len(res))
    print('The Total MSE is {}'.format(total_RMSE))
