import os
import shutil
import random

import numpy as np
import torch
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

class ImageProcess:
    def get_one_hot_labels(self, data_dir):
        # 加载数据集
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 你可以根据需要调整图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(data_dir, transform=transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # 获取类别名称和类别到标签的映射
        classes = dataset.classes
        class_to_idx = dataset.class_to_idx
        num_classes = len(classes)

        print("Classes:", classes)
        print("Class to index mapping:", class_to_idx)

        # 生成独热编码标签
        one_hot_labels = []

        for _, label in data_loader:
            label = label.item()  # 获取标签索引
            one_hot = torch.nn.functional.one_hot(torch.tensor(label), num_classes)
            one_hot_labels.append(one_hot.numpy())

        # 将列表转换为一个大的 NumPy 矩阵
        one_hot_matrix = np.vstack(one_hot_labels)
        return one_hot_matrix

        return one_hot_labels
    @staticmethod
    def split_images(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        # 获取所有图片文件的列表
        all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        # 打乱文件列表
        random.shuffle(all_images)

        # 计算每个文件夹应包含的文件数
        total_images = len(all_images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count

        # 创建目标文件夹
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 复制文件到各个文件夹
        for i, image in enumerate(all_images):
            src_path = os.path.join(source_dir, image)
            if i < train_count:
                dest_path = os.path.join(train_dir, image)
            elif i < train_count + val_count:
                dest_path = os.path.join(val_dir, image)
            else:
                dest_path = os.path.join(test_dir, image)
            shutil.copy(src_path, dest_path)

        print(f"Processed {total_images} images from {source_dir}")
        print(f"Train images: {train_count}")
        print(f"Validation images: {val_count}")
        print(f"Test images: {test_count}")

    @staticmethod
    def process_all_subfolders(parent_dir, output_dir_base, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        # 获取所有子文件夹的列表
        subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

        for subfolder in subfolders:
            # 检查子文件夹是否包含更多子文件夹
            inner_subfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
            if inner_subfolders:
                # 递归处理内部子文件夹
                ImageProcess.process_all_subfolders(subfolder, output_dir_base, train_ratio, val_ratio, test_ratio)
            else:
                category_name = os.path.basename(subfolder)

                # 创建对应的训练、验证和测试文件夹
                train_dir = os.path.join(output_dir_base, 'train', category_name)
                val_dir = os.path.join(output_dir_base, 'val', category_name)
                test_dir = os.path.join(output_dir_base, 'test', category_name)

                # 处理当前子文件夹中的图片
                ImageProcess.split_images(subfolder, train_dir, val_dir, test_dir, train_ratio, val_ratio, test_ratio)


if __name__ == "__main__":
    source_directory = 'data/RSI-CB128-after_delete'
    output_directory = 'data/output'
    test_data = 'data/output/test'
    ImageProcess = ImageProcess()
    ImageProcess.process_all_subfolders(source_directory, output_directory)
    data_dir = 'data/output/test'  # 指定你的数据文件夹路径
    one_hot_matrix = ImageProcess.get_one_hot_labels(data_dir)

    print("One-Hot Matrix Shape:", one_hot_matrix.shape)
    print(one_hot_matrix)

#  还未添加图像变化处理如旋转等
