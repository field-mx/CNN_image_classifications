import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
print('f')
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []  # 存储图像文件路径
        self.labels = []       # 存储标签

        # 读取标签 CSV 文件
        labels_csv = os.path.join(data_dir, 'airport_runway_train_labels.csv')
        df = pd.read_csv(labels_csv)

        # 遍历标签 CSV 文件中的每一行，获取图像文件路径和标签
        for idx, row in df.iterrows():
            image_name = row['image']  # 图像文件名
            label = row['label']        # 图像标签
            image_path = os.path.join(data_dir, image_name)
            self.image_paths.append(image_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像文件
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 如果定义了转换方法，则对图像进行转换
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
