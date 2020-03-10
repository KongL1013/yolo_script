# -*- coding: utf-8 -*-
# @Author: lushujie
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from configer import configer

sets=[('final', 'train'), ('final', 'val'), ('final', 'test')]
config = configer()
classes = config.classes


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(final, image_id):
    curent_xml_path = os.path.join(config.xml_path,str(image_id)+'.xml')
    in_file = open(curent_xml_path,'r')
    out_file = open('%s/%s.txt'%(config.labels_path,image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def voc_label():
    print("start voc_label")
    for final, image_set in sets:
        if not os.path.exists(config.labels_path):
            os.makedirs(config.labels_path)
        image_ids = open('%s/%s.txt'%(config.txt_path,image_set)).read().strip().split()
        list_file = open('%s/%s_%s.txt'%(config.txt_path,final, image_set), 'w')
        for image_id in image_ids:
            current_pic_path = os.path.join(config.pic_path,str(image_id)+'.'+config.pic_type+'\n')
            list_file.write(current_pic_path)
            convert_annotation(final, image_id)
        list_file.close()

if __name__ == '__main__':
    voc_label()