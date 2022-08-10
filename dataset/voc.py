import os
import random


ROOT = os.path.dirname(os.path.abspath(__file__))

 
trainval_percent = 0.9  # 训练和验证集所占比例，剩下的0.1就是测试集的比例
train_percent = 0.8  # 训练集所占比例，可自己进行调整

src_path = ROOT + '/data_process/'

xmlfilepath = src_path + 'VOCdevkit/VOC2007/Annotations'
txtsavepath = src_path + 'VOCdevkit/VOC2007/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
# print(total_xml)
num = len(total_xml)
list = range(num)
# print(list)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
ftrainval = open(src_path + 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
ftest = open(src_path + 'VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
ftrain = open(src_path + 'VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
fval = open(src_path + 'VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    # print(name)
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
 