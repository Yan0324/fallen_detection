# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import Namespace
import argparse

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def is_fallen(keypoints, thr_angle=45, thr_confidence=0.3):
    """基于关键点判断是否跌倒
    Args:
        keypoints (np.ndarray): [N,3] 关键点数组（x,y,score）
        thr_angle (float): 判定跌倒的角度阈值（单位：度）
        thr_confidence (float): 关键点置信度阈值
    Returns:
        bool: 是否跌倒
    """
    # 关键点索引（COCO格式）
    NECK = 1
    LHIP = 11
    RHIP = 12
    LKNEE = 13
    RKNEE = 14
    
    # 获取关键点坐标（注意：图像坐标系Y轴向下）
    neck = keypoints[NECK]
    l_hip = keypoints[LHIP]
    r_hip = keypoints[RHIP]
    
    # 置信度检查
    if neck[2] < thr_confidence or l_hip[2] < thr_confidence or r_hip[2] < thr_confidence:
        return False
    
    # 计算髋部中点
    hip_center = ((l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2)
    
    # 计算身体中线向量（从髋部中心指向颈部）
    body_vector = (neck[0] - hip_center[0], neck[1] - hip_center[1])
    
    # 计算与垂直方向（图像Y轴）的夹角
    vertical_vector = (0, 1)  # 图像坐标系Y轴向下
    cos_theta = np.dot(body_vector, vertical_vector) / (
        np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
    # 检查膝盖位置（跌倒时髋部低于膝盖）
    if keypoints[LKNEE][2] > thr_confidence and keypoints[RKNEE][2] > thr_confidence:
        avg_knee_y = (keypoints[LKNEE][1] + keypoints[RKNEE][1]) / 2
        if hip_center[1] > avg_knee_y:  # Y坐标越大位置越靠下
            return False
    
    return angle > thr_angle

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    return data_samples.get('pred_instances', None), visualizer.get_image()


def main():
    """Hardcoded configuration for video pose estimation."""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Fall Detection from Video')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    args_cmd = parser.parse_args()

    # 硬编码所有参数
    args = Namespace(
        # 使用Faster RCNN作为默认检测器（需提前下载权重）
        det_config='mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py',
        det_checkpoint='checkpoint/detect/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
        
        # 使用HRNet作为姿态估计模型（需提前下载权重）
        pose_config='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py',
        pose_checkpoint='checkpoint/pose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
        
        # 输入输出设置
        input=args_cmd.video,      # 使用视频文件
        show=True,           # 实时显示结果
        output_root='outputs',  # 输出目录
        save_predictions=False,
        
        # 硬件设置
        device='cuda:0',     # 使用GPU
        
        # 检测参数
        det_cat_id=0,        # 人体检测类别ID（COCO数据集中0代表人）
        bbox_thr=0.3,        # 检测框置信度阈值
        nms_thr=0.3,         # NMS阈值
        
        # 可视化参数
        kpt_thr=0.3,         # 关键点显示阈值
        draw_heatmap=False,  # 是否绘制热力图
        show_kpt_idx=False,  # 是否显示关键点索引
        skeleton_style='mmpose',
        radius=3,            # 关键点半径
        thickness=1,         # 骨架线条粗细
        show_interval=0,     # 显示间隔
        alpha=0.8,           # 检测框透明度
        draw_bbox=True       # 绘制检测框
    )

    assert has_mmdet, '请先安装mmdet'

    # 准备输出路径
    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, 'video_output.mp4')

    # 初始化检测器
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # 初始化姿态估计器
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # 初始化可视化工具
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    # 开启视频文件
    cap = cv2.VideoCapture(args.input)
    video_writer = None
    frame_idx = 0

    # 初始化报警状态
    fall_duration = 0
    ALARM_DURATION = 5  # 持续5帧触发报警

    print(f"开始处理视频文件: {args.input}，按ESC退出...")
    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1
        if not success:
            break

        # 姿态估计
        pred_instances, frame_vis = process_one_image(args, frame, detector,
                                           pose_estimator, visualizer, 0.001)

        # 处理跌倒检测
        if pred_instances is not None:
            keypoints = pred_instances.keypoints
            scores = pred_instances.keypoint_scores
            
            for i in range(len(keypoints)):
                # 组合坐标和置信度
                kpts = np.concatenate([keypoints[i], scores[i][:, None]], axis=1)
                
                # 打印关键点信息用于调试
                print(f"KeyPoints: {kpts}")
                
                if is_fallen(kpts):
                    fall_duration += 1
                    print(f"Fall Detected (Duration: {fall_duration}/{ALARM_DURATION})")
                    if fall_duration >= ALARM_DURATION:
                        # 绘制报警提示
                        cv2.putText(frame_vis, "FALL DETECTED!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    fall_duration = max(0, fall_duration - 1)

        # 处理输出
        if output_file:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    25,  # fps
                    (frame_vis.shape[1], frame_vis.shape[0]))
            video_writer.write(mmcv.rgb2bgr(frame_vis))

        # 实时显示
        if args.show:
            cv2.imshow('Fall Detection', mmcv.rgb2bgr(frame_vis))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # 释放资源
    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if output_file:
        print(f'处理后的视频已保存至: {output_file}')


if __name__ == '__main__':
    main()