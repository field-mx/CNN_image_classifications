import os
import shutil
import random
import pandas as pd

print('1')
class ImageProcess:
    def __init__(self):
        pass

    def create_labels_csv(self, filenames, labels, output_csv):
        # 创建包含文件名和标签的 DataFrame
        df = pd.DataFrame({'filename': filenames, 'label': labels})

        # 保存 DataFrame 为 CSV 文件
        df.to_csv(output_csv, index=False)

    @staticmethod
    def split_images(source_dir, train_dir, val_dir, test_dir, train_csv, val_csv, test_csv, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
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

        train_files, val_files, test_files = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        # 复制文件到各个文件夹
        for i, image in enumerate(all_images):
            src_path = os.path.join(source_dir, image)
            if i < train_count:
                dest_path = os.path.join(train_dir, image)
                train_files.append(image)
                train_labels.append(os.path.basename(source_dir))
            elif i < train_count + val_count:
                dest_path = os.path.join(val_dir, image)
                val_files.append(image)
                val_labels.append(os.path.basename(source_dir))
            else:
                dest_path = os.path.join(test_dir, image)
                test_files.append(image)
                test_labels.append(os.path.basename(source_dir))
            shutil.copy(src_path, dest_path)

        # 创建 CSV 文件
        ImageProcess().create_labels_csv(train_files, train_labels, train_csv)
        ImageProcess().create_labels_csv(val_files, val_labels, val_csv)
        ImageProcess().create_labels_csv(test_files, test_labels, test_csv)

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

                # 创建 CSV 文件路径
                train_csv = os.path.join(output_dir_base, 'train', category_name, f'{category_name}_train_labels.csv')
                val_csv = os.path.join(output_dir_base, 'val', category_name, f'{category_name}_val_labels.csv')
                test_csv = os.path.join(output_dir_base, 'test', category_name, f'{category_name}_test_labels.csv')

                # 处理当前子文件夹中的图片
                ImageProcess.split_images(subfolder, train_dir, val_dir, test_dir, train_csv, val_csv, test_csv, train_ratio, val_ratio, test_ratio)


if __name__ == "__main__":
    source_directory = 'data/RSI-CB128-after_delete'
    output_directory = 'data/output'
    ImageProcess.process_all_subfolders(source_directory, output_directory)
