

import numpy as np
import matplotlib.pyplot as plt
import random

import copy

path = './fish_s_osnet_ibn_x1_0_MSMT176/tracks/test.txt'


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


def dist(a, b):
    return np.sqrt(sum((a - b) ** 2))

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内
'''
fish = dict()
keys = list(data_dict.keys())
for i in range(len(keys)):
    data_lines = data_dict[keys[i]]
    data_line_list = []
    for j in range(len(data_lines)):
        data_line_list.append(data_lines[j][1:6])
    for k in range(len(data_line_list)):
        
        if()
        
        
        fish[data_lines[j][1]] = 
'''

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

data_6 = dict()
'''
data_6['1'] = data_dict_2['1']
data_6['2'] = data_dict_2['2']
data_6['3'] = data_dict_2['3']
data_6['4'] = data_dict_2['4']
data_6['5'] = data_dict_2['5']
data_6['6'] = data_dict_2['6']


keys = list(data_dict_2.keys())

for i in range(len(keys)):
    
    data_line_list = [data_6['1'][-1][2:6],data_6['2'][-1][2:6],data_6['3'][-1][2:6],data_6['4'][-1][2:6],data_6['5'][-1][2:6],data_6['6'][-1][2:6]]
    
    res_cos_list = []
    
    for j in range(len(data_line_last)):
        
        
        
        x_vec = [int(x) for x in data_dict_2[str(keys[i])][0][2:6]]
        y_vec = [int(y) for y in data_line_last[j]]
        
        
        res_cos =  dist(np.array(x_vec),np.array(y_vec))
        res_cos_list.append(res_cos)
    
    index = res_cos_list.index(min(res_cos_list)) + 1
    data_6[str(index)] = data_6[str(index)] + data_dict_2[str(keys[i])]
    
    
    

interval = 2

'''


data_6['1'] = []
data_6['2'] = []
data_6['3'] = []
data_6['4'] = []
data_6['5'] = []
data_6['6'] = []
keys = list(data_dict.keys())
res_dict = copy.deepcopy(data_dict)
count = 0
count_del = 0
for i in range(len(keys)):

    res_data_line = copy.deepcopy(res_dict[keys[i]])
    res_data_line_list = []
    res_data_line_id_list = []
    res_names = ['1','2','3','4','5','6']




    if(len(res_data_line) > 6):
        count_del = count_del + 1
        while (True):
            del (res_data_line[-1])
            del (res_dict[keys[i]][-1])
            if (len(res_data_line) == 6):
                break


    for k in range(len(res_data_line)):
        res_data_line_list.append(res_data_line[k])
        res_data_line_id_list.append(res_data_line[k][1])
    #res_modify = [x for x in res_data_line_list]
    for n in range(len(res_data_line_id_list)):
        if(res_data_line_id_list[n] in res_names):
            inx = res_names.index(res_data_line_id_list[n])
            res_names[inx] = -1
            res_data_line_list[n][1] = -1
            res_data_line_id_list[n] = -1
    for v in range(len(res_data_line_list)):
        if(res_data_line_list[v][1]!=-1 and res_names!=[-1,-1,-1,-1,-1,-1] and int(res_data_line_list[v][1])>6):

            tmp_res_names = []
            for a in range(len(res_names)):
                if(res_names[a]!=-1):
                    tmp_res_names.append(res_names[a])
                    if(len(tmp_res_names) != 1):
                        count = count + 1
                        print('count', count)
            for c in range(i,len(keys)):
                for t in range(len(res_dict[keys[c]])):
                    if(res_dict[keys[c]][t][1] == res_data_line_list[v][1]):
                        if(len(tmp_res_names) == 1):
                            res_dict[keys[c]][t][1] = tmp_res_names[0]
                        else:
                            res_dict[keys[c]][t][1] = tmp_res_names[-1]
                            #print('others')
                if (res_data_line_list[v][1] == '54'):
                    print('stop')
    if(i%10==0):
        print(i)

print('count_del',count_del)
for i in range(len(keys)):
    res_data_line = res_dict[keys[i]]

    for k in range(len(res_data_line)):
        data_6[res_data_line[k][1]].append(res_data_line[k][2:6])
        if(int(res_data_line[k][1]) > 7):
            print('stop')
    for j in range(len(data_6)):
        keys_6 = list(data_6.keys())
        for n in range(len(keys_6)):
            if(len(data_6[keys_6[n]])<(i+1)):
                while(True):
                    data_6[keys_6[n]].append([0,0,0,0])
                    if(len(data_6[keys_6[n]]) == (i+1)):
                        break
            elif(len(data_6[keys_6[n]])>(i+1)):
                while(True):
                    del(data_6[keys_6[n]][-1])
                    if(len(data_6[keys_6[n]]) == (i + 1)):
                        break
'''
for i in range(len(keys) - 1):

    res_data_line = data_dict[keys[i]]
    res_data_line_list = []
    res_data_line_id_list = []
    res_names = ['1','2','3','4','5','6']
    for k in range(len(res_data_line)):
        res_data_line_list.append(res_data_line[k])
        res_data_line_id_list.append(res_data_line[k][1])
    #res_modify = [x for x in res_data_line_list]
    for n in range(len(res_data_line_id_list)):
        if(res_data_line_id_list[n] in res_names):
            inx = res_names.index(res_data_line_id_list[n])
            res_names[inx] = -1
            res_data_line_list[n][1] = -1
            res_data_line_id_list[n] = -1
    for v in range(len(res_data_line_list)):
        if(res_data_line_list[v][1]!=-1):
            for c in range(len(keys)):
                if(res_dict[keys[c]][1] == res_data_line_list[v][1]):
                    res_dict[keys[c]][1] = res_names[v]
    #data_line_last = [data_6['1'][-1][2:6], data_6['2'][-1][2:6], data_6['3'][-1][2:6], data_6['4'][-1][2:6],
                      #data_6['5'][-1][2:6], data_6['6'][-1][2:6]]

    res_cos_list = []

    ddd = data_dict[keys[i+1]]
    if(len(ddd)<6):
        while(True):
            ddd.append(['0','0','0','0','0','0'])
            if(len(ddd)==6):
                break
    elif(len(ddd)>6):
        while (True):
            del(ddd[-1])
            if (len(ddd) == 6):
                break
    data_line_last = []
    data_line_last_id = []
    for k in range(len(ddd)):
        data_line_last.append(ddd[k][2:6])
        data_line_last_id.append(ddd[k][1])

    ccc = data_dict[keys[i]]
    if (len(ccc) > 6):
        while (True):
            del(ccc[-1])
            if (len(ccc) == 6):
                break
    elif (len(ccc) < 6):
        while (True):
            ccc.append(['0','0','0','0','0','0'])
            if (len(ccc) == 6):
                break
    data_line_first = []
    data_line_first_id = []
    for q in range(len(ccc)):
        data_line_first.append(ccc[q][2:6])
        data_line_first_id.append(ccc[q][1])

    for v in range(len(data_line_last)):
        res_dis_list = []
        if (data_line_last_id[v] in data_line_first_id):
            try:
                #index = data_line_first_id.index(data_line_first_id[v]) + 1
                data_6[str(v+1)].append(data_line_last[v])
                continue
            except:
                print('error')

        for j in range(len(data_line_first)):

            x_vec = [int(x) for x in data_line_last[v]]
            y_vec = [int(y) for y in data_line_first[j]]
            res_dis = dist(np.array(x_vec), np.array(y_vec))
            res_dis_list.append(res_dis)
        index = res_dis_list.index(min(res_dis_list)) + 1

        data_6[str(index)].append(data_line_last[index-1])
    keys_2 = list(data_6.keys())
    for o in range(len(keys_2)):
        if(len(data_6[keys_2[o]]) == (i+1) ):
            continue
        elif(len(data_6[keys_2[o]]) > (i+1) ):
            while(True):
                del(data_6[keys_2[o]][-1])
                if(len(data_6[keys_2[o]])==(i+1)):
                    break
        elif (len(data_6[keys_2[o]]) < (i + 1)):
            while (True):
                data_6[keys_2[o]].append([])
                if (len(data_6[keys_2[o]]) == (i + 1)):
                    break

for i in range(1,len(data_6['1'])):
    if (data_6['1'][i] == []):
        data_6['1'][i] = data_6['1'][i - 1]
    if (data_6['2'][i] == []):
        data_6['2'][i] = data_6['2'][i - 1]
    if (data_6['3'][i] == []):
        data_6['3'][i] = data_6['3'][i - 1]
    if (data_6['4'][i] == []):
        data_6['4'][i] = data_6['4'][i - 1]
    if (data_6['5'][i] == []):
        data_6['5'][i] = data_6['5'][i - 1]
    if (data_6['6'][i] == []):
        data_6['6'][i] = data_6['6'][i - 1]
'''

print('end')
'''

#!/usr/bin/env python
import cv2
cap=cv2.VideoCapture("C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\test.mp4")

if cap.isOpened():
    ret,frame=cap.read()
else:
    ret = False
n=0
i=0
timeF = 1
path='C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\img\\'+'/{}'
while ret:
    n = n + 1
    ret,frame=cap.read()
    if (n%timeF == 0) :
        i = i+1
        print(i)
        filename=str(n)+"_"+str(i)+".jpg"
        cv2.imwrite(path.format(filename),frame)

cap.release()
'''
'''
import cv2

count = count + 1
flag = [0,0,0,0,0,0]
for i in range(len(data_6['1'])):
    if(i%100==0):
        print('cv2.rec:',i)
    im_det = cv2.imread("C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\img\\"+str(i+1)+'_'+str(i+1)+'.jpg')
    for j in range(1,7):
        cv2.rectangle(im_det, (int(data_6[str(j)][i][0]), int(data_6[str(j)][i][1])), (int(data_6[str(j)][i][0])+int(data_6[str(j)][i][2]), int(data_6[str(j)][i][1])+int(data_6[str(j)][i][3])), (0, 0, 255), 2)
        cv2.putText(im_det, 'fish '+ str(j), (int(data_6[str(j)][i][0]), int(data_6[str(j)][i][1]) - 5), 0, 0.6, (0, 0, 255), 2)

        if(180<int(data_6[str(j)][i][1])<360 and flag[j-1] == 1):
            flag[j - 1] = 0
        elif((int(data_6[str(j)][i][1]) > 360 or int(data_6[str(j)][i][1]) < 180) and flag[j - 1] == 0):
            flag[j - 1] = 1
            count = count + 1
    cv2.putText(im_det, 'count ' + str(j), (850, int(data_6[str(j)][i][1]) - 5), 0, 0.6,(0, 0, 255), 2)
    cv2.imwrite("C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\img_det\\"+str(i+1)+'_'+str(i+1)+'.jpg',im_det)

'''
'''
import cv2
seconds = 3
count = 0
flag = dict()
flag_count = dict()
dict2_keys = list(data_dict_2.keys())
for i in range(len(dict2_keys)):
    flag[dict2_keys[i]] = 0
    flag_count[dict2_keys[i]] = 0
dict_keys = list(data_dict.keys())
for i in range(31*seconds,len(dict_keys)):
    if(i%100==0):
        print('cv2.rec:',i)
    im_det = cv2.imread("C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\img\\"+str(i+1)+'_'+str(i+1)+'.jpg')
    data_line_key = data_dict[dict_keys[i]]
    for j in range(len(data_line_key)):
        light = 0
        flag_count[data_line_key[j][1]] = flag_count[data_line_key[j][1]] + 1
        if(180<int(data_line_key[j][3])<360 and flag[data_line_key[j][1]] == 1):
            flag[data_line_key[j][1]] = 0
        elif(int(data_line_key[j][3]) > 360 and flag[data_line_key[j][1]] == 0):
            time_flag = 0
            for p in range(1,31*seconds):
                try:
                    if(int(data_dict_2[data_line_key[j][1]][flag_count[data_line_key[j][1]]-p][3]) < 360):
                        time_flag = 1
                        break
                except:
                    break
            if(time_flag == 0):
                flag[data_line_key[j][1]] = 1
                count = count + 1
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
                    break
            if (time_flag == 0):
                flag[data_line_key[j][1]] = 1
                count = count + 1
                light = 1
                cv2.rectangle(im_det, (int(data_line_key[j][2]), int(data_line_key[j][3])), (int(data_line_key[j][2]) + int(data_line_key[j][4]),int(data_line_key[j][3]) + int(data_line_key[j][5])), (0, 255, 0), 2)
                cv2.putText(im_det, 'fish ' + str(data_line_key[j][1]),(int(data_line_key[j][2]), int(data_line_key[j][3]) - 5), 0, 0.6, (0, 255, 0), 2)
        if(light == 0):
            cv2.rectangle(im_det, (int(data_line_key[j][2]), int(data_line_key[j][3])), (int(data_line_key[j][2]) + int(data_line_key[j][4]), int(data_line_key[j][3]) + int(data_line_key[j][5])),(0, 0, 255), 1)
            cv2.putText(im_det, 'fish ' + str(data_line_key[j][1]),(int(data_line_key[j][2]), int(data_line_key[j][3]) - 5), 0, 0.6, (0, 0, 255), 1)

    im_det[179:180,:,:] = (0, 255, 0)
    im_det[359:360, :, :] = (0, 255, 0)
    cv2.putText(im_det, 'count ' + str(count), (850, 270), 0, 0.6,(0, 0, 255), 2)
    cv2.imwrite("C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\img_det\\"+str(i+1)+'_'+str(i+1)+'.jpg',im_det)

'''

#!/usr/bin/env python
import cv2

img = cv2.imread('C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\img_det\\94_94.jpg')
imginfo = img.shape
size = (imginfo[1],imginfo[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWrite = cv2.VideoWriter('C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\7.mp4',fourcc,30,size)

for i in range(len(data_6['1'])):
    filename = 'C:\\Users\\cmcc\\Desktop\\Yolov5_StrongSORT_OSNet-5.0\\img_det\\'+str(i+1)+'_'+str(i+1)+'.jpg'
    img = cv2.imread(filename,1)
    videoWrite.write(img)
    print(i)

videoWrite.release()
print('end')


    
'''

    data_interval = []
    for j in range(interval):
        data_interval.append(data_dict[keys[i+j]])
    
    for k in range(len(data_interval[0])):
        for q in range(len(data_interval[1])):
            
    
    
    break
    
    
'''

