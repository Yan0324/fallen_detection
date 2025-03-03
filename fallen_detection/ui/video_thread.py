import sys
import os
sys.path.append('../../../mmpose')
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer
import torch
import numpy as np

class VideoProcessor(QThread):
    frame_processed = pyqtSignal(QImage)
    
    def __init__(self, config_path, checkpoint_path):
        super().__init__()

        # 增加检测模型初始化
        self.det_config = '../../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'
        self.det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth'
        
        # 初始化检测模型
        self.det_model = init_model(
            self.det_config, 
            self.det_checkpoint,
            device='cuda:0'
        )

        self.mutex = QMutex()
        register_all_modules()
        self.model = init_model(config_path, checkpoint_path, device='cuda:0')
        self.visualizer = PoseLocalVisualizer()
        if hasattr(self.model, 'dataset_meta'):
            self.visualizer.set_dataset_meta(
                self.model.dataset_meta, skeleton_style='mmpose'
            )
        self.running = False
        self.cap = None

    def run(self):
        self.mutex.lock()
        self.running = True
        self.cap = cv2.VideoCapture(0)
        # 设置摄像头的宽度和高度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)  # 设置宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 设置高度
        self.mutex.unlock()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    results = inference_topdown(self.model, frame)
                    
                    # 可视化处理
                    self.visualizer.add_datasample(
                        'result',
                        frame,
                        data_sample=results[0],
                        show=False,
                        draw_gt=False,
                        draw_bbox=True
                    )
                    vis_frame = self.visualizer.get_image()
                    
                    # 转换图像格式
                    rgb_image = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    qt_image = QImage(
                        rgb_image.data, 
                        w, h, 
                        QImage.Format.Format_RGB888
                    )
                    self.frame_processed.emit(qt_image)
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()