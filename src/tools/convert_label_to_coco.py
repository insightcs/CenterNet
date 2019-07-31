#coding: utf-8

import os
import sys
import json
import cv2
import glob
import numpy as np
from tqdm import tqdm
from lxml import etree

def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
    Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

class Label2COCO(object):
    def __init__(self, images_dir, labels_dir, output_path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.xml_list = glob.glob(os.path.join(labels_dir, '*.xml'))
        self.output_path = output_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.classes_name = ['chair', 'car', 'horse', 'person', 'bicycle', 'cat', 'dog',
                             'train', 'aeroplane', 'diningtable', 'tvmonitor', 'bird', 'bottle',
                             'motorbike', 'pottedplant', 'boat', 'sofa', 'sheep', 'cow', 'bus']

        for idx, class_name in enumerate(self.classes_name):
            self.categories.append({'supercategory': 'none', 'id': idx+1, 'name': class_name})

        self.save_json()

    def data_transfer(self):
        ann_id = 0
        for num, xml_file in enumerate(tqdm(self.xml_list)):

            xml_file = os.path.join(self.labels_dir, xml_file)
            with open(xml_file, 'rb') as file_reader:
                xml_str = file_reader.read().decode('utf-8')
            xml = etree.fromstring(xml_str)
            xml_data = recursive_parse_xml_to_dict(xml)['annotation']
            img_name = xml_data['filename']
            img_path = os.path.join(self.images_dir, img_name)
            if not os.path.exists(img_path):
                print('image {}: not exists'.format(img_name))
                exit(0)
            height = xml_data['size']['height']
            width = xml_data['size']['width']
            self.images.append(self.image(height, width, num+1, img_name))

            if 'object' in xml_data:
                for obj in xml_data['object']:
                    class_name = obj['name']
                    if class_name not in self.classes_name:
                        print('class name {} not exists'.format(class_name))
                        exit(0)

                    xmin = int(float(obj['bndbox']['xmin']))
                    xmax = int(float(obj['bndbox']['xmax']))
                    ymin = int(float(obj['bndbox']['ymin']))
                    ymax = int(float(obj['bndbox']['ymax']))
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]   # COCO 对应格式[x,y,w,h]
                    self.annotations.append(self.annotation([], bbox, class_name, num+1, ann_id+1))
                    ann_id += 1
            else:
                print('image {}: not exists'.format(xml_file))
                exit(0)

    def image(self, img_h, img_w, img_id, file_name):
        image = {}
        image['height'] = img_h
        image['width'] = img_w
        image['id'] = img_id
        image['file_name'] = file_name
        return image

    def categorie(self, class_name):
        categorie = {}
        categorie['supercategory'] = 'none'
        categorie['id'] = len(self.classes_name) + 1  # 0 默认为背景
        categorie['name'] = class_name
        return categorie

    def annotation(self, img_seg, bbox, class_name, img_id, ann_id):
        annotation = {}
        # annotation['segmentation'] = [self.getsegmentation()]
        annotation['segmentation'] = [img_seg]
        annotation['iscrowd'] = 0
        annotation['image_id'] = img_id
        annotation['bbox'] = bbox
        annotation['category_id'] = self.getcatid(class_name)
        annotation['id'] = ann_id
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        with open(self.output_path, 'w') as file_writer:
            json.dump(self.data_coco, file_writer, indent=4)  # indent=4 更加美观显示


if __name__ == '__main__':
    images_dir = ''
    labels_dir = ''
    output_path = ''
    Label2COCO(images_dir, labels_dir, output_path)