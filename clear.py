import os


def clear_csv_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")


def remove_empty_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # Check if the directory is empty
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty folder: {dir_path}")
                except Exception as e:
                    print(f"Failed to remove folder {dir_path}. Reason: {e}")


if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "data/output"

    # 清除文件夹及所有子文件夹中的所有.csv文件
    clear_csv_files(folder_path)

    # 清除所有空文件夹
    remove_empty_folders(folder_path)
