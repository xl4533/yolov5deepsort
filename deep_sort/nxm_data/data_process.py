import os
from PIL import Image
from shutil import copyfile, copytree, rmtree, move
import cv2

PATH_DATASET = './nxm_data/crops/fish'  # 需要处理的文件夹
PATH_NEW_DATASET = './nxm_data/stitches'  # 处理后的文件夹
PATH_ALL_IMAGES = PATH_NEW_DATASET + '/all_images'
PATH_TRAIN = PATH_NEW_DATASET + '/train'
PATH_TEST = PATH_NEW_DATASET + '/test'

src_dir = PATH_DATASET


for root,dir1,filename in os.walk(src_dir,True):
    for index in range(len(filename)):
        if(os.path.splitext(filename[index])[1]=='.jpg'):
            file_index = filename[index].split('.')[0].split('_')[-1]
            img = cv2.imread(root + '/' + filename[index])
            cv2.imwrite('./nxm_data/images/'+file_index+'/'+filename[index],img)