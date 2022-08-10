cp Arial.ttf /root/.config/Ultralytics/
python ./dataset/voc.py
python ./dataset/xml_2_txt.py
rm -rf ./yolov5/VOCdevkit
cp -r ./dataset/data_process/VOCdevkit ./yolov5/