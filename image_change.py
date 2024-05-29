from PIL import Image
import os
#这是一个把图片压缩为200*150的代码，与本项目无关

def resize_image(input_path, size):
    """
    Resize the input image to the specified size and save it to the same directory with a new name.

    :param input_path: Path to the input image.
    :param size: Tuple indicating the target size (width, height).
    """
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.Resampling.LANCZOS)

        # 分离文件名和扩展名
        dir_name, file_name = os.path.split(input_path)
        base_name, ext = os.path.splitext(file_name)

        # 创建新文件名
        new_file_name = f"{base_name}_resized{ext}"
        output_path = os.path.join(dir_name, new_file_name)

        # 保存调整大小后的图像
        resized_img.save(output_path)

        return output_path


if __name__ == "__main__":
    input_image_path = "C:/Users/32009/Pictures/Saved Pictures/mx.jpg"  # 输入图像路径
    target_size = (150, 200)  # 目标大小 (宽度, 高度)

    output_image_path = resize_image(input_image_path, target_size)
    print(f"Image resized and saved to {output_image_path}")
