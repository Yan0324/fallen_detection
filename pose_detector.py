# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms

class PoseDetector:
    """实时姿态检测器，支持报警信息叠加"""
    
    def __init__(self, config):
        # 初始化模型
        self.detector = init_detector(
            config.det_config, 
            config.det_checkpoint, 
            device=config.device
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        
        self.pose_estimator = init_model(
            config.pose_config,
            config.pose_checkpoint,
            device=config.device
        )
        
        # 可视化参数
        self.show = config.show
        self.draw_bbox = config.draw_bbox
        self.alarm_status = False

    def process_frame(self, frame, alert_callback=None):
        """处理单帧并返回带标注的图像"""
        # 人体检测
        det_result = inference_detector(self.detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = self._filter_bboxes(pred_instance)
        
        # 姿态估计
        pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
        
        # 报警回调（处理所有实例）
        all_keypoints = []
        all_scores = []
        for result in pose_results:
            all_keypoints.append(result.pred_instances.keypoints)
            all_scores.append(result.pred_instances.keypoint_scores)
        
        if alert_callback and len(pose_results) > 0:
            self.alarm_status = alert_callback(all_keypoints, all_scores)
        
        # 可视化处理
        frame_vis = self._visualize(frame, pose_results)
        return frame_vis
    
    # def _filter_bboxes(self, pred_instance, 
    #                 det_cat_id=0, bbox_thr=0.3, nms_thr=0.3):
    #     """过滤检测框（带NMS）"""
    #     bboxes = np.concatenate(
    #         (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    #     bboxes = bboxes[np.logical_and(
    #         pred_instance.labels == det_cat_id,
    #         pred_instance.scores > bbox_thr)]
        
    #     if len(bboxes) < 1:
    #         return np.zeros((0, 4))
        
    #     keep = nms(bboxes[:, :4], bboxes[:, -1], nms_thr)
    #     return bboxes[keep][:, :4]
    
    def _filter_bboxes(self, pred_instance, 
                det_cat_id=0, bbox_thr=0.3, nms_thr=0.3):
        """过滤检测框（带NMS）"""
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(
            pred_instance.labels == det_cat_id,
            pred_instance.scores > bbox_thr)]
        
        if len(bboxes) < 1:
            return np.zeros((0, 4))
        
        # 新版NMS调用方式（根据MMPose 1.0+调整）
        keep = nms(bboxes, nms_thr)
        return bboxes[keep][:, :4]



    def _visualize(self, frame, pose_results):
        """可视化处理（含报警叠加）"""
        frame_vis = mmcv.bgr2rgb(frame)
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)
        
        # 绘制报警状态
        if self.alarm_status:
            cv2.putText(frame_vis, "FALL ALERT!", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.rectangle(frame_vis, (20, 20), (620, 100), (0, 0, 255), 3)
            
        # 绘制所有检测框和关键点
        if self.draw_bbox:
            for result in pose_results:
                for bbox in result.pred_instances.bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        return frame_vis
