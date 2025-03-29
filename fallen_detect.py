# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import Namespace

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

    return data_samples.get('pred_instances', None)


def main():
    """Hardcoded configuration for webcam pose estimation."""
    # 硬编码所有参数
    args = Namespace(
        # 使用Faster RCNN作为默认检测器
        det_config='mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py',
        det_checkpoint='checkpoint/detect/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
        
        # 使用RTMPose作为姿态估计模型
        pose_config='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py',
        pose_checkpoint='checkpoint/pose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
        
        # 输入输出设置
        input='webcam',      # 使用摄像头
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
        output_file = os.path.join(args.output_root, 'webcam_output.mp4')

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

    # 开启摄像头
    cap = cv2.VideoCapture(0)
    video_writer = None
    frame_idx = 0

    print("开始实时姿态估计，按ESC退出...")
    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1
        if not success:
            break

        # 姿态估计
        pred_instances = process_one_image(args, frame, detector,
                                           pose_estimator, visualizer, 0.001)

        # 处理输出
        if output_file:
            frame_vis = visualizer.get_image()
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    25,  # fps
                    (frame_vis.shape[1], frame_vis.shape[0]))
            video_writer.write(mmcv.rgb2bgr(frame_vis))

        # 实时显示
        if args.show and cv2.waitKey(5) & 0xFF == 27:
            break

    # 释放资源
    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if output_file:
        print(f'视频已保存至: {output_file}')


if __name__ == '__main__':
    main()
