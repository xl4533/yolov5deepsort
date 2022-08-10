import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
 
#sets设置的就是
sets=['train', 'val', 'test']
 
 
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["fish"]  # 修改为自己的label
 
def convert(size, box):
    dw = 1./(size[0])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dw = 1./((size[0])+0.1)
    dh = 1./(size[1])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dh = 1./((size[0])+0.1)
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(image_id):
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml'%(image_id))
    out_file = open('VOCdevkit/VOC2007/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()
 
for image_set in sets:
    if not os.path.exists('VOCdevkit/VOC2007/labels/'):  # 修改路径（最好使用全路径）
        os.makedirs('VOCdevkit/VOC2007/labels/')  # 修改路径（最好使用全路径）
    image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()  # 修改路径（最好使用全路径）
    list_file = open('VOCdevkit/VOC2007/%s.txt' % (image_set), 'w')  # 修改路径（最好使用全路径）
    for image_id in image_ids:
        list_file.write('VOCdevkit/VOC2007/JPEGImages/%s.jpg\n' % (image_id))  # 修改路径（最好使用全路径）
        convert_annotation(image_id)
    list_file.close()
 