# video_thread.py 修改后的完整代码
import sys
import os
sys.path.append('../../../mmpose')
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage
import numpy as np
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
# MMPose相关导入
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import merge_data_samples
# MMDetection相关导入
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
from fall_detection import FallDetection

class VideoProcessor(QThread):
    frame_processed = pyqtSignal(QImage)
    fall_detected = pyqtSignal()  # 新增信号
    
    def __init__(self, pose_config, pose_checkpoint, det_config, det_checkpoint):
        super().__init__()
        self.mutex = QMutex()
        register_all_modules()

        
        
        # 初始化检测模型（YOLOX示例，可按需更换）
        self.detector = init_detector(
            det_config,
            det_checkpoint,
            device='cuda:0'
        )
        
        # 初始化姿态估计模型
        self.pose_model = init_model(
            pose_config,
            pose_checkpoint,
            device='cuda:0'
        )
        
        # 初始化可视化工具
        self.visualizer = PoseLocalVisualizer()
        if hasattr(self.pose_model, 'dataset_meta'):
            self.visualizer.set_dataset_meta(
                self.pose_model.dataset_meta, skeleton_style='mmpose'
            )
        
        self.running = False
        self.cap = None
        self.det_score_thr = 0.5  # 检测置信度阈值
        self.nms_thr = 0.3       # NMS阈值
        self.fall_detector = FallDetection()  # 初始化摔倒检测器

    def run(self):
        self.mutex.lock()
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.mutex.unlock()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Step 1: 人体检测
                init_default_scope('mmdet')
                det_result = inference_detector(self.detector, frame)
                
                # 过滤检测结果（假设检测类别0是人）
                pred_instances = det_result.pred_instances.cpu().numpy()
                bboxes = np.concatenate(
                    (pred_instances.bboxes, pred_instances.scores[:, None]), axis=1)
                valid_indices = np.where(
                    (pred_instances.labels == 0) & 
                    (pred_instances.scores > self.det_score_thr)
                )[0]
                bboxes = bboxes[valid_indices]
                
                # 应用NMS（非极大值抑制）
                if len(bboxes) > 0:
                    from mmpose.evaluation.functional import nms
                    bboxes = bboxes[nms(bboxes, self.nms_thr)]
                # Step 2: 多人姿态估计
                if len(bboxes) > 0:
                    pose_results = inference_topdown(
                        self.pose_model, 
                        frame, 
                        bboxes=bboxes[:, :4],  # 保留前四列坐标
                        bbox_format='xyxy'
                    )
                    data_samples = merge_data_samples(pose_results)
                else:
                    #创建空的数据样本避免None
                    data_samples = PoseDataSample()
                    data_samples.pred_instances = InstanceData()  # 确保包含pred_instances属性
                    pose_results = []  # 初始化空列表

                
                # 可视化结果
                self.visualizer.add_datasample(
                    'result',
                    frame,
                    data_sample=data_samples,
                    draw_gt=False,
                    draw_bbox=True,
                    draw_heatmap=False,
                    show=False
                )
                
                vis_frame = self.visualizer.get_image()

                # if not pose_results or len(pose_results[0].pred_instances.keypoints) < 17:
                #     continue

                # 检测是否摔倒
                if self.fall_detector.is_fall(pose_results):
                    self.fall_detected.emit()  # 发射信号
                    self.emit_alarm()
                
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
    #报警函数
    
    def emit_alarm(self):
        """
        发出摔倒报警，这里可以是音频、UI警示或者其他方式
        """
        
        print("警报！检测到摔倒！") 