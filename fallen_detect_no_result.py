# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmengine import Registry
from mmengine import Config
from mmengine.logging import print_log
from argparse import Namespace

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def main():
    # 硬编码配置参数
    args = Namespace(
        # 检测模型配置（需提前下载权重）
        det_config='mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        det_checkpoint='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        
        # 使用RTMPose作为姿态估计模型
        pose_config='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py',
        pose_checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth',
        
        # 运行参数
        device='cuda:0',    # 使用GPU加速
        show=True,          # 启用实时显示
        det_cat_id=0,       # 人体检测类别ID
        bbox_thr=0.3,       # 检测框置信度阈值
        nms_thr=0.3,        # NMS阈值
        kpt_thr=0.3,        # 关键点显示阈值
        draw_bbox=True,     # 绘制检测框
        skeleton_style='mmpose'  # 骨骼连接样式
    )

    # 初始化人体检测器
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # 初始化姿态估计模型
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )

    # 初始化可视化工具
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, 
        skeleton_style=args.skeleton_style
    )

    # 开启摄像头
    cap = cv2.VideoCapture(0)
    print("开启摄像头，按ESC键退出...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 人体检测
        det_result = inference_detector(detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        
        # 处理检测结果
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                       pred_instance.scores > args.bbox_thr)]
        bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

        # 姿态估计
        pose_results = inference_topdown(pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)

        # 可视化处理
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        visualizer.add_datasample(
            'result',
            img_rgb,
            data_sample=data_samples,
            draw_gt=False,
            draw_bbox=args.draw_bbox,
            kpt_thr=args.kpt_thr,
            show=False)
        
        # 实时显示
        vis_frame = cv2.cvtColor(visualizer.get_image(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Real-time Pose Estimation', vis_frame)

        # ESC键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
