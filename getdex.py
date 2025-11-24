from fileinput import filename
from math import sqrt
from androguard.core import apk
import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from androguard.util import set_log
set_log("ERROR")

# 现在，当你使用Androguard时，它应该不会再输出debug信息了

path = "/Data2/masy/CIC/CICMAL"
out_path = "./data/image_CICMAL"
files = os.listdir(path)
out_files = os.listdir(out_path)
# dexl = []

# 加载APK文件
for apkfile in files:
    file_name = os.path.splitext(apkfile)[0]   
    image_name = file_name + ".png"
    if image_name in out_files:
        continue
    print(file_name)
    try:
        apk1 = apk.APK(path + '/' + apkfile)
    except Exception as e:
            os.remove(os.path.join(path,apkfile))
            print(e)
            exit(0)
    
    dex_bytes = b''.join(apk1.get_all_dex())
    try:
        arr = np.frombuffer(dex_bytes, dtype=np.uint16)
    except:
        print(len(dex_bytes))
        arr = np.frombuffer(dex_bytes[:-1], dtype=np.uint16)
    if arr.size < 512*512:
        print(file_name)
        # arr = np.pad(arr, (0, 65536 - arr.size), mode='constant', constant_values=0)
        try:
            repeat_factor = 512*512 // arr.size + 1
        except Exception as e:
            os.remove(os.path.join(path,apkfile))
            print(e)
            exit(0)
        arr = np.tile(arr, repeat_factor)
    # else:
    #     continue
    squ = int(sqrt(arr.size))
    arr = arr[:squ * squ]
    np_array = arr.reshape(squ,squ)
    zoom_factor = 512 / squ
    compressed_array = zoom(np_array, zoom_factor, order=3)

    image_zoom = Image.fromarray(compressed_array, mode='I;16')

    image_zoom.save(os.path.join(out_path, image_name))
