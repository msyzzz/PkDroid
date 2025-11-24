import os
import zipfile
import sys
import pack_list
import argparse
import pandas as pd
from androguard.core.axml import AXMLPrinter
from androguard.util import set_log
from tqdm import tqdm

try:
    set_log("ERROR")
except ValueError:
    pass  # 没有 handler，忽略

def parse_args():
    parser = argparse.ArgumentParser(description="查壳")
    parser.add_argument('--path', type=str, required=True,
                        help='输入APK样本目录')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='查壳结果输出文件路径(csv文件)')
    return parser.parse_args()


def get_api(xmldata):
    xml = AXMLPrinter(xmldata).get_xml().decode("utf-8")
    for key, value in pack_list.API.items():
        for api in value:
            if api in xml:
                return key
            else:
                return "未识别加固"

def get_pack(app_path):
    folder_path, file_name = os.path.split(app_path)
    try:
        unzip = zipfile.ZipFile(app_path)
    except:
        return "安装包格式不正确"
    else:
        names = unzip.namelist()
        for i in names:
            for key, value in pack_list.SO.items():
                for so in value:
                    if i.find(so) > 0:
                        return key
        xmldata = unzip.read("AndroidManifest.xml")
        pack = get_api(xmldata)
        return pack
        # return "未识别加固"


def main():
    args = parse_args()
    input_dir = args.path
    output_name = args.output
    if not output_name.lower().endswith('.csv'):
        print("错误：日志文件必须以 .csv 结尾")
        sys.exit(1)

    df = pd.DataFrame(columns=["name", "packname"])
    apk_list = os.listdir(input_dir)
    for i in tqdm(apk_list, desc="Processing apks"):
        apk_path = os.path.join(input_dir, i)
        apk_name = i.split(".")[0]
        pack = get_pack(apk_path)
        df.loc[len(df)] = [apk_name, pack]
    
    df.to_csv(output_name, index=False)


if __name__ == "__main__":
    main()