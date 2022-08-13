pip install xlwt
pip install xlrd
pip install xlutils
rm -rf ./dataset/data_process/video_image_det
mkdir ./dataset/data_process/video_image_det
rm -rf ./inference/output/*
cd runs/track/
python calcaulate.py