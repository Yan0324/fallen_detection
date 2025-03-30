# config.py
class Config:
    def __init__(self):
        # 模型参数
        self.det_config = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'
        self.det_checkpoint = 'checkpoint/detect/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
        self.pose_config = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py'
        self.pose_checkpoint = 'checkpoint/pose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'
        self.device = 'cuda:0'
        
        # 可视化参数
        self.show = True
        self.draw_bbox = True
        self.bbox_thr = 0.3     # 检测框置信度阈值
        self.nms_thr = 0.3      # NMS阈值
        self.radius = 4
        self.alpha = 0.8
        self.output_root = 'outputs'
        self.det_cat_id=0,       # 人体检测类别ID      
        self.kpt_thr=0.3,        # 关键点显示阈值
        self.draw_bbox=True,     # 绘制检测框
        self.skeleton_style='mmpose'  # 骨骼连接样式

