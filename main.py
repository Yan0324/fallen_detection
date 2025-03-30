import cv2
from argparse import Namespace
from pose_detector import PoseDetector
from fallen_alert import FallDetector
from config import Config

def main():
    # 配置参数（可抽离为单独配置文件）
    # config = Namespace(
    #     # 检测模型配置
    #     # det_config='configs/faster_rcnn.py',
    #     # det_checkpoint='checkpoints/faster_rcnn.pth',
    #     # pose_config='configs/rtmpose.py',
    #     # pose_checkpoint='checkpoints/rtmpose.pth',
    #     # 使用Faster RCNN作为默认检测器
    #     det_config='mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py',
    #     det_checkpoint='checkpoint/detect/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
    #     # 使用RTMPose作为姿态估计模型
    #     pose_config='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py',
    #     pose_checkpoint='checkpoint/pose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
    #     device='cuda:0'
    # )
    config = Config()
    # 初始化模块
    pose_detector = PoseDetector(config)
    fall_detector = FallDetector(
        angle_thr=45,
        confidence_thr=0.3,
        min_duration=5
    )
    
    # 开启摄像头
    cap = cv2.VideoCapture(0)
    print("启动跌倒检测系统，按ESC退出...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧数据（通过回调传递报警判断）
        processed_frame = pose_detector.process_frame(
            frame, 
            alert_callback=fall_detector.check_fall
        )
        
        # 显示结果
        cv2.imshow('Fall Detection', processed_frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("系统已关闭")

if __name__ == '__main__':
    main()
