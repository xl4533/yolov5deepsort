


import cv2
import argparse

def threshold_By_OTSU(input_img_file):
    image=input_img_file
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
    ret, binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)#自定义阈值
    #print("threshold value %s" % ret)  #打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    return binary

def video2frame(videos_path,frames_save_path,time_interval):

    '''
      :param videos_path: 视频的存放路径
      :param frames_save_path: 视频切分成帧之后图片的保存路径
      :param time_interval: 保存间隔
      :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        #image = threshold_By_OTSU(image)
        count += 1
        if count % time_interval == 0:
            cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
        if count % 1000 == 0: 
            print(count)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='dataset/data_source/input.mp4', help='source')
    parser.add_argument('--dst', type=str, default='dataset/data_process/video_image', help='dst')
    parser.add_argument('--interval', type=str, default=1, help='time interval')
    opt = parser.parse_args()
    videos_path = opt.source
    frames_save_path = opt.dst
    time_interval = opt.interval#隔一帧保存一次
    video2frame(videos_path, frames_save_path, time_interval)
