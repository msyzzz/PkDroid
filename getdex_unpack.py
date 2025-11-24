import os
import argparse
import numpy as np
from math import sqrt
from PIL import Image
from scipy.ndimage import zoom
from androguard.core import dex
from androguard.util import set_log
from tqdm import tqdm
set_log("ERROR")


def parse_args():
    parser = argparse.ArgumentParser(description="将解包目录中的.dex文件生成16位灰度图")
    parser.add_argument('--path', type=str, default='./unpack',
                        help='脱壳后DEX文件夹所在目录（默认为 ./unpack）')
    parser.add_argument('--output', '-o', type=str, default='./data/train/image_unpack',
                        help='输出图像文件的目录（默认为 ./data/train/image_unpack）')
    return parser.parse_args()


def get_all_apk_folders(input_dir):
    return [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]


def has_valid_dex(folder_path):
    for name in os.listdir(folder_path):
        if name.endswith(".dex"):
            return True
    return False


def collect_valid_dex_bytes(folder_path, min_size=50 * 1024):
    """
    从文件夹中读取并拼接所有合法（大小合适，能被解析）的 .dex 文件的二进制内容。
    min_size：用于筛去过短的加固相关DEX
    """
    dex_bytes_list = []
    for name in os.listdir(folder_path):
        if not name.endswith(".dex"):
            continue
        full_path = os.path.join(folder_path, name)
        if os.path.getsize(full_path) < min_size:
            continue
        with open(full_path, 'rb') as f:
            dex_data = f.read()
            try:
                _ = dex.DEX(dex_data)
            except Exception:
                pass
            dex_bytes_list.append(dex_data)
    return b''.join(dex_bytes_list)


def convert_bytes_to_image_array(dex_bytes):
    """
    将 dex 字节流转换为 numpy 图像数组，压缩为 512x512 并返回。
    """
    try:
        array = np.frombuffer(dex_bytes, dtype=np.uint16)
    except Exception:
        return None
    if array.size == 0:
        return None
    required = 512 * 512
    if array.size < required:
        repeat = required // array.size + 1
        array = np.tile(array, repeat)
    squ = int(sqrt(array.size))
    array = array[:squ * squ]
    matrix = array.reshape(squ, squ)
    scale = 512 / squ
    resized = zoom(matrix, scale, order=3)
    return resized


def save_image(array, output_path):
    image = Image.fromarray(array, mode='I;16')
    image.save(output_path)


def process_folder(folder_name, input_dir, output_dir, existing_images):
    image_name = folder_name + ".png"
    
    if image_name in existing_images:
        return
    folder_path = os.path.join(input_dir, folder_name)
    if not has_valid_dex(folder_path):
        return
    
    dex_bytes = collect_valid_dex_bytes(folder_path)
    tqdm.write(image_name)
    if not dex_bytes:
        return
    tqdm.write(f"[processing] {folder_name}, dex size: {len(dex_bytes)}")
    image_array = convert_bytes_to_image_array(dex_bytes)
    if image_array is None:
        tqdm.write(f"[skip] {folder_name}: 无法转换")
        return
    image_path = os.path.join(output_dir, image_name)
    save_image(image_array, image_path)
    tqdm.write(f"[saved] {image_path}")


def main():
    args = parse_args()
    input_dir = args.path
    output_dir = args.output
    folders = get_all_apk_folders(input_dir)
    existing = set(os.listdir(output_dir))
    for folder in tqdm(folders, desc="Processing Folders"):
        process_folder(folder, input_dir, output_dir, existing)


if __name__ == "__main__":
    main()
