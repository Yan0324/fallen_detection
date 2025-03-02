import sys
import os
sys.path.append('../../mmpose')
from mmpose.apis import MMPoseInferencer

import cv2
import torch
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import adapt_mmdet_pipeline

# 配置参数
config_file = '../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py'  # 模型配置文件路径
checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth'  # 预训练权重
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 优先使用GPU
show_fps = True  # 显示FPS

# 初始化模型
model = init_model(config_file, checkpoint_file, device=device)

# 启动摄像头
cap = cv2.VideoCapture(0)
assert cap.isOpened(), '摄像头初始化失败'

# 新版 MMPose 数据格式修复
# 修改 process_results 函数以处理多人
def process_results(results):
    all_keypoints = []
    all_scores = []
    for result in results:
        instances = result.pred_instances
        all_keypoints.extend(instances.keypoints)
        all_scores.extend(instances.keypoint_scores)
    return all_keypoints, all_scores


connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 手臂
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16) # 腿部
]

print("按 'q' 退出实时检测...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # 推理
    results = inference_topdown(model, frame)
    
    # 处理结果
    keypoints, scores = process_results(results)
    
    if keypoints is not None:
        # 绘制关键点
        for i, (x, y) in enumerate(keypoints):
            if scores[i] > 0.3:
                cv2.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)
        
        # 绘制骨架
        for (start, end) in connections:
            if start < len(scores) and end < len(scores):
                if scores[start] > 0.3 and scores[end] > 0.3:
                    x1, y1 = keypoints[start].astype(int)
                    x2, y2 = keypoints[end].astype(int)
                    cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 2)
    
    # 显示FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()