import matplotlib

matplotlib.use('Agg')  # 设置matplotlib使用Agg后端
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/yolo11.yaml')
    # model.load('yolov11n.pt') # loading pretrain weights
    model.train(data='/home/dataset/SIMD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='1',
                optimizer='SGD',
                resume=True,
                project='runs/train',
                name='exp',
                amp=False, 
                )
