#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
import argparse

from Cython.Compiler.Naming import self_cname
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from tif_process import *
from slice_dataset import slice

# root path for saving the tif and shp file.
ROOT = r'./example_data/original_data'
# ROOT = 'D:/DOWNLO/shp2coco_master/example_data/original_data'
img_path = 'D:/DOWNLO/building file/sliceShp/111/img'
shp_path = 'D:/DOWNLO/building file/sliceShp/111/shp'
# root path for saving the mask.
ROOT_DIR = ROOT + '/dataset'
IMAGE_DIR = os.path.join(ROOT_DIR, "emcv_2024")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

clip_size = 512

INFO = {
    "description": "EMCV Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2024,
    "contributor": "HXF",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'building',
        'supercategory': 'building',
    },
{
        'id': 2,
        'name': 'road',
        'supercategory': 'road',
    },
{
        'id': 3,
        'name': 'waterbody',
        'supercategory': 'waterbody',
    },
{
        'id': 4,
        'name': 'forest',
        'supercategory': 'forest',
    },
# {
#         'id': 5,
#         'name': 'grassland',
#         'supercategory': 'grassland',
#     },
# {
#         'id': 6,
#         'name': 'denseforest',
#         'supercategory': 'denseforest',
#     },
# {
#         'id': 7,
#         'name': 'sparseforest',
#         'supercategory': 'sparseforest',
#     },
]

def filter_for_jpeg(root, files):# 筛选出tif格式的图像
    # file_types = ['*.jpeg', '*.jpg']
    file_types = ['*.tiff', '*.tif']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):# 筛选出与tif图像名称相同的标签文件
    # file_types = ['*.png']
    file_types = ['*.tif']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    # file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    # files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    # 筛选出与图像文件的基本名称前缀匹配的文件
    files = [f for f in files if basename_no_extension == os.path.splitext(os.path.basename(f))[0].split('_', 1)[0]]

    return files

def from_mask_to_coco(root, MARK, IMAGE, ANNOTATION):
    ROOT_DIR = root + '/' + MARK
    IMAGE_DIR = ROOT_DIR + '/' + IMAGE
    ANNOTATION_DIR = ROOT_DIR + '/' + ANNOTATION
    if os.path.exists(ROOT_DIR):
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        image_id = 1
        segmentation_id = 1

        # filter for jpeg images使用filter_for_jpeg过滤图像
        for root, _, files in os.walk(IMAGE_DIR):
            image_files = filter_for_jpeg(root, files)

            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                # 使用 pycococreatortools.create_image_info 创建图像的 COCO 信息，并将其添加到 coco_output["images"] 列表中
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                # filter_for_annotations 是一个自定义函数，用于从标注目录中筛选出与当前图像文件关联的标注文件
                for root, _, files in os.walk(ANNOTATION_DIR):
                    annotation_files = filter_for_annotations(root, files, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        # 读取标注文件，获取类别 ID（通过与 CATEGORIES 中的类别名称匹配）
                        print(annotation_filename)
                        class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename)
                                                 .convert('1')).astype(np.uint8)

                        # 使用 pycococreatortools.create_annotation_info 创建标注信息，并将其添加到 coco_output["annotations"] 列表中
                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)


                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1
        # 将生成的 COCO 格式数据（coco_output 字典）保存为 JSON 文件
        with open('{}/instances_{}2024.json'.format(ROOT_DIR, MARK), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
    else:
        print(ROOT_DIR + ' does not exit!')

def main():
    # clip_from_file(clip_size, ROOT, img_path, shp_path)
    # slice(ROOT_DIR, train=0.8, eval=0.1, test=0.1)
    # from_mask_to_coco(ROOT_DIR, 'train', "emcv_2024", "annotations")
    # from_mask_to_coco(ROOT_DIR, 'eval', "emcv_2024", "annotations")
    # from_mask_to_coco(ROOT_DIR, 'test', "emcv_2024", "annotations")
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', nargs='+',
                        choices=['clip', 'split', 'coco_all', 'coco_train', 'coco_eval', 'coco_test'],
                        help='''执行步骤:
                           clip - 裁剪数据
                           split - 划分数据集
                           coco_all - 生成全部标注
                           coco_train - 仅生成训练集标注
                           coco_eval - 仅生成验证集标注
                           coco_test - 仅生成测试集标注''')

    args = parser.parse_args()

    if not args.step:
        print("未指定执行步骤，请使用--step参数选择操作")
        return

    # 保持原始函数调用方式完全不变
    if 'clip' in args.step:
        print("执行数据裁剪...")
        clip_from_file(clip_size, ROOT, img_path, shp_path)

    if 'split' in args.step:
        print("划分数据集...")
        slice(ROOT_DIR, train=0.8, eval=0.1, test=0.1)

    if 'coco_all' in args.step:
        print("生成全部COCO标注...")
        from_mask_to_coco(ROOT_DIR, 'train', "emcv_2024", "annotations")
        from_mask_to_coco(ROOT_DIR, 'eval', "emcv_2024", "annotations")
        from_mask_to_coco(ROOT_DIR, 'test', "emcv_2024", "annotations")
    else:
        if 'coco_train' in args.step:
            from_mask_to_coco(ROOT_DIR, 'train', "emcv_2024", "annotations")
        if 'coco_eval' in args.step:
            from_mask_to_coco(ROOT_DIR, 'eval', "emcv_2024", "annotations")
        if 'coco_test' in args.step:
            from_mask_to_coco(ROOT_DIR, 'test', "emcv_2024", "annotations")
if __name__ == "__main__":
    main()
