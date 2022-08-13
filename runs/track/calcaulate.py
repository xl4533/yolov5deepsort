

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import copy
import time



def TimeStampToTime(timestamp):
    timeStruct = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)

def get_FileCreateTime(filePath):
    # 获取文件的创建时间
    # filePath = unicode(filePath,'utf8')
    t = os.path.getctime(filePath)
    return TimeStampToTime(t)

src_dir = './'
time_list = []
src_list = []
for root,dir1,filename in os.walk(src_dir,True):
    for index in range(len(filename)):
        if(os.path.splitext(filename[index])[1]=='.txt'):
            time_list.append(get_FileCreateTime(root + '/' + filename[index]))
            src_list.append(root + '/' + filename[index])
index = time_list.index(max(time_list))
path = src_list[index]


with open(path,'r',encoding='utf-8') as f:
    file = f.read()

data = file.split('\n')

data_dict = dict()

for i in range(len(data)):
    if(data[i]==''):
        continue
    
    data_line = data[i].split(' ')
    try:
        data_dict[data_line[0]].append(data_line)
    except:
        data_dict[data_line[0]] = []
        data_dict[data_line[0]].append(data_line)

data_dict_2 = dict()

for i in range(len(data)):
    if(data[i]==''):
        continue
    data_line = data[i].split(' ')
    try:
        data_dict_2[data_line[1]].append(data_line)
    except:
        data_dict_2[data_line[1]] = []
        data_dict_2[data_line[1]].append(data_line)

import cv2
seconds = 3
count = 0
count_top = 0
count_middle = 0
count_bottom = 0
flag = dict()
flag_count = dict()
dict2_keys = list(data_dict_2.keys())
for i in range(len(dict2_keys)):
    flag[dict2_keys[i]] = 1
    flag_count[dict2_keys[i]] = 0
dict_keys = list(data_dict.keys())

count_index = len(dict_keys) / 60*30 + 1
count_min_top = [0]
count_min_middle = [0]
count_min_bottom = [0]

for i in range(31*seconds,len(dict_keys)):
    if(i%1800==0):
        count_min_top.append(count_top - sum(count_min_top))
        count_min_middle.append(count_middle - sum(count_min_middle))
        count_min_bottom.append(count_bottom - sum(count_min_bottom))
        print('cv2.rec:',i)
    #if(i%36000 == 0):
    #    break
    im_det = cv2.imread("../../dataset/data_process/video_image/frame"+str(i+1)+".jpg")
    data_line_key = data_dict[dict_keys[i]]
    for j in range(len(data_line_key)):
        light = 0
        flag_count[data_line_key[j][1]] = flag_count[data_line_key[j][1]] + 1
        if(180<int(data_line_key[j][3])<360 and flag[data_line_key[j][1]] == 1):
            flag[data_line_key[j][1]] = 0
            time_flag = 0
            for p in range(1, 31 * seconds):
                try:
                    if ((int(data_dict_2[data_line_key[j][1]][flag_count[data_line_key[j][1]] - p][3]) > 360) or (int(data_dict_2[data_line_key[j][1]][flag_count[data_line_key[j][1]] - p][3]) < 180)):
                        time_flag = 1
                        break
                except:
                    time_flag = 1
                    break
            if (time_flag == 0):
                count_middle = count_middle + 1
                light = 1
                cv2.rectangle(im_det, (int(data_line_key[j][2]), int(data_line_key[j][3])), (int(data_line_key[j][2]) + int(data_line_key[j][4]),int(data_line_key[j][3]) + int(data_line_key[j][5])), (0, 255, 0), 2)
                cv2.putText(im_det, 'fish ' + str(data_line_key[j][1]),(int(data_line_key[j][2]), int(data_line_key[j][3]) - 5), 0, 0.6, (0, 255, 0), 2)

        elif(int(data_line_key[j][3]) > 360 and flag[data_line_key[j][1]] == 0):
            time_flag = 0
            for p in range(1,31*seconds):
                try:
                    if(int(data_dict_2[data_line_key[j][1]][flag_count[data_line_key[j][1]]-p][3]) < 360):
                        time_flag = 1
                        break
                except:
                    time_flag = 1
                    break
            if(time_flag == 0):
                flag[data_line_key[j][1]] = 1
                count = count + 1
                count_bottom = count_bottom + 1
                light = 1
                cv2.rectangle(im_det, (int(data_line_key[j][2]), int(data_line_key[j][3])), (int(data_line_key[j][2]) + int(data_line_key[j][4]),int(data_line_key[j][3]) + int(data_line_key[j][5])), (0, 255, 0), 2)
                cv2.putText(im_det, 'fish ' + str(data_line_key[j][1]),(int(data_line_key[j][2]), int(data_line_key[j][3]) - 5), 0, 0.6, (0, 255, 0), 2)
        elif(int(data_line_key[j][3]) < 180  and flag[data_line_key[j][1]] == 0 ):
            time_flag = 0
            for p in range(1, 31 * seconds):
                try:
                    if (int(data_dict_2[data_line_key[j][1]][flag_count[data_line_key[j][1]]-p][3]) > 180):
                        time_flag = 1
                        break
                except:
                    time_flag = 1
                    break
            if (time_flag == 0):
                flag[data_line_key[j][1]] = 1
                count = count + 1
                count_top = count_top + 1
                light = 1
                cv2.rectangle(im_det, (int(data_line_key[j][2]), int(data_line_key[j][3])), (int(data_line_key[j][2]) + int(data_line_key[j][4]),int(data_line_key[j][3]) + int(data_line_key[j][5])), (0, 255, 0), 2)
                cv2.putText(im_det, 'fish ' + str(data_line_key[j][1]),(int(data_line_key[j][2]), int(data_line_key[j][3]) - 5), 0, 0.6, (0, 255, 0), 2)
        if(light == 0):
            cv2.rectangle(im_det, (int(data_line_key[j][2]), int(data_line_key[j][3])), (int(data_line_key[j][2]) + int(data_line_key[j][4]), int(data_line_key[j][3]) + int(data_line_key[j][5])),(0, 0, 255), 1)
            cv2.putText(im_det, 'fish ' + str(data_line_key[j][1]),(int(data_line_key[j][2]), int(data_line_key[j][3]) - 5), 0, 0.6, (0, 0, 255), 1)

    im_det[179:180,:,:] = (0, 255, 0)
    im_det[359:360, :, :] = (0, 255, 0)
    cv2.putText(im_det, 'count_top ' + str(count_top), (750, 90), 0, 0.6,(0, 0, 255), 2)
    cv2.putText(im_det, 'count_middle ' + str(count_middle), (750, 270), 0, 0.6, (0, 0, 255), 2)
    cv2.putText(im_det, 'count_bottom ' + str(count_bottom), (750, 450), 0, 0.6, (0, 0, 255), 2)
    cv2.imwrite("../../dataset/data_process/video_image_det/frame"+str(i+1)+".jpg",im_det)



#!/usr/bin/env python
import cv2

img = cv2.imread('../../dataset/data_process/video_image_det/frame100.jpg')
imginfo = img.shape
size = (imginfo[1],imginfo[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWrite = cv2.VideoWriter('../../inference/output/out.mp4',fourcc,30,size)

for i in range(len(dict_keys)):
    filename = "../../dataset/data_process/video_image_det/frame"+str(i+1)+".jpg"
    img = cv2.imread(filename,1)
    videoWrite.write(img)
    if(i%1000==0):
        print(i)

videoWrite.release()

print('count_min_top',count_min_top)
print('count_min_middle',count_min_middle)
print('count_min_bottom',count_min_bottom)

'''

    data_interval = []
    for j in range(interval):
        data_interval.append(data_dict[keys[i+j]])
    
    for k in range(len(data_interval[0])):
        for q in range(len(data_interval[1])):
            
    
    
    break
    
    
'''
import xlwt
import xlrd
from xlutils.copy import copy

workbook = xlwt.Workbook(encoding='utf-8')
# 创建一个worksheet
worksheet = workbook.add_sheet('My Worksheet')
# 写入excel
# 参数对应 行, 列, 值goldsilverplatinumpalladium
worksheet.write(0, 0, label='时间(分钟)')
worksheet.write(0, 1, label='上层区域')
worksheet.write(0, 2, label='中间区域')
worksheet.write(0, 3, label='下层区域')
# 保存
workbook.save("../../inference/output/out.xls")

# 打开需要操作的excel表
wb = xlrd.open_workbook("../../inference/output/out.xls")
# 复制原有表
newb = copy(wb)

# 获取原有excel表中sheet名为‘My Worksheet’的sheet
sumsheet = newb.get_sheet('My Worksheet')
# k表示该sheet的最后一行
k = len(sumsheet.rows) - 1
# 想原有sheet后面新增数据
# 参数对应 行, 列, 值
for i in range(1,len(count_min_top)):
    sumsheet.write(k + i, 0, label=i)
    sumsheet.write(k + i, 1, label=count_min_top[i])
    sumsheet.write(k + i, 2, label=count_min_middle[i])
    sumsheet.write(k + i, 3, label=count_min_bottom[i])
# 保存
newb.save("../../inference/output/out.xls")


print('end')
