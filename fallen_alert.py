import numpy as np

class FallDetector:
    """跌倒检测器，支持状态持续跟踪"""
    
    def __init__(self, angle_thr=45, confidence_thr=0.3, min_duration=5):
        self.ANGLE_THR = angle_thr
        self.CONF_THR = confidence_thr
        self.MIN_DURATION = min_duration
        self.fall_duration = 0
        self.is_alert = False
        
    def check_fall(self, all_keypoints, all_scores):
        """检查所有实例是否跌倒"""
        any_fallen = False
        for keypoints, scores in zip(all_keypoints, all_scores):
            # 确保输入数据的正确形状
            if keypoints.ndim == 3:
                keypoints = keypoints[0]  # 去掉批次维度 (1,17,2) → (17,2)
            if scores.ndim == 2:
                scores = scores[0]        # 去掉批次维度 (1,17) → (17,)
                
            kpts = np.concatenate([
                keypoints, 
                scores[:, None]
            ], axis=1)  # shape (17,3)
            
            if self._is_fallen(kpts):
                any_fallen = True
                break
        return any_fallen

    def _is_fallen(self, kpts):
        """修正后的核心判断逻辑"""
        # COCO正确索引定义
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        
        # 颈部是左右肩的中点
        neck = (kpts[LEFT_SHOULDER] + kpts[RIGHT_SHOULDER]) / 2
        l_hip = kpts[LEFT_HIP]
        r_hip = kpts[RIGHT_HIP]
        
        # 置信度检查（每个关键点的第三位是score）
        conf_check = (
            neck[2] > self.CONF_THR and 
            l_hip[2] > self.CONF_THR and 
            r_hip[2] > self.CONF_THR
        )
        if not conf_check:
            return False
        
        # 计算髋部中心
        hip_center = (l_hip[:2] + r_hip[:2]) / 2
        
        # 身体向量（从髋部中心到颈部）
        body_vector = neck[:2] - hip_center
        
        # 计算与垂直方向的夹角
        vertical = np.array([0, 1])
        cos_theta = np.dot(body_vector, vertical) / (
            np.linalg.norm(body_vector) * np.linalg.norm(vertical))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        
        # 检查膝盖高度
        if (kpts[LEFT_KNEE][2] > self.CONF_THR and 
            kpts[RIGHT_KNEE][2] > self.CONF_THR):
            avg_knee_y = (kpts[LEFT_KNEE][1] + kpts[RIGHT_KNEE][1]) / 2
            if hip_center[1] > avg_knee_y:
                return False
                
        return angle > self.ANGLE_THR
    
    def _trigger_alarm(self):
        print("\n[ALERT] 检测到人员跌倒！")
        
    def _stop_alarm(self):
        print("[INFO] 警报解除")
