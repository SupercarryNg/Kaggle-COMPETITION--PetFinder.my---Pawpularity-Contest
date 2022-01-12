import os.path
import pandas as pd
import timm
from torch.utils.data import Dataset
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# # 设定IMAGE SIZE >>> 缩放比例， 设定剪裁尺寸
# IMAGE_SIZE = 224
#
# # 获得图片转换Tensor的实例化
# imgTransform = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE),             		  # 比例缩放至合适尺寸，将图片短边缩放至x，长宽比保持不变
#     transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 裁剪合适大小的图像
#     transforms.ToTensor(),                            # 转换成Tensor形式
#     transforms.Normalize(mean=(0.5, 0.5, 0.5),        # Normalization
#                          std=(0.5, 0.5, 0.5))
#     ])

model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
config = resolve_data_config({}, model=model)
imgTransform = create_transform(**config)


class MyData(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.csv_path = os.path.join(self.root_dir, self.sub_dir) + '.csv'
        tmp = pd.read_csv(self.csv_path)[['Id', 'labels']]
        self.popularity = tmp
        self.path = os.path.join(self.root_dir, self.sub_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        id = img_name[:-4]
        img_item_path = os.path.join(self.root_dir, self.sub_dir, img_name)
        img = Image.open(img_item_path).convert('RGB')
        img = imgTransform(img)
        pawpularity = self.popularity.loc[self.popularity.Id == id, 'labels'].values
        return img, pawpularity

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    rootdir = 'petfinder-pawpularity-score'
    subdir = 'train'
    traindataset = MyData(rootdir, subdir)
    img, p = traindataset[981]
    print(img.size())
    print('type of img is {}'.format(type(img)))
    print(p)
    print('type of p is {}'.format(type(p)))

