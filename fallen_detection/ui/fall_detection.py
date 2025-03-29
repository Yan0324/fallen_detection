import numpy as np
from collections import deque
from mmengine.structures import InstanceData

class FallDetection:
    def __init__(self, 
                 angle_threshold=45,
                 height_ratio_threshold=0.35,
                 min_confidence=0.4,
                 head_hip_threshold=0.25,
                 aspect_ratio_threshold=1.8,
                 horizontal_threshold=0.15,
                 velocity_threshold=0.08,
                 min_consecutive_frames=3,
                 history_size=5):
        """综合多维度检测参数"""
        self.angle_threshold = angle_threshold
        self.height_ratio_threshold = height_ratio_threshold
        self.min_confidence = min_confidence
        self.head_hip_threshold = head_hip_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.horizontal_threshold = horizontal_threshold
        self.velocity_threshold = velocity_threshold
        self.min_consecutive_frames = min_consecutive_frames
        
        # 轨迹分析队列
        self.history = deque(maxlen=history_size)
        self.consecutive_frames = 0
        self.last_detection_params = {}

    def _get_valid_point(self, point, img_size=1.0):
        """带归一化处理的坐标点获取"""
        if np.isnan(point[0]) or np.isnan(point[1]):
            return (np.nan, np.nan)
        return (point[0]/img_size, point[1]/img_size)

    def _calculate_velocity(self, current_centroid):
        """计算运动速度"""
        if len(self.history) < 2:
            return 0
        dx = current_centroid[0] - self.history[-2][0]
        dy = current_centroid[1] - self.history[-2][1]
        return np.sqrt(dx**2 + dy**2)

    def calculate_spine_angle(self, keypoints, img_height):
        """改进的脊柱角度计算"""
        # 获取并验证关键点
        def get_point(idx):
            point = keypoints[idx]
            return (
                point[0]/img_height if not np.isnan(point[0]) else np.nan,
                point[1]/img_height if not np.isnan(point[1]) else np.nan
            )

        left_hip = get_point(11)
        right_hip = get_point(12)
        left_shoulder = get_point(5)
        right_shoulder = get_point(6)

        # 详细验证每个坐标分量
        if any(np.isnan(coord) for point in [left_hip, right_hip, left_shoulder, right_shoulder] for coord in point):
            return np.nan

        # 计算中间点
        mid_hip = ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2)
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0])/2, 
                    (left_shoulder[1] + right_shoulder[1])/2)

        # 计算脊柱向量
        spine_vector = np.array([mid_shoulder[0] - mid_hip[0], 
                            mid_shoulder[1] - mid_hip[1]])
        
        # 计算与垂直方向的夹角
        vertical_vector = np.array([0, 1])
        if np.linalg.norm(spine_vector) < 1e-6:
            return 0.0
        
        cosine = np.dot(spine_vector, vertical_vector) / (
            np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
        return np.degrees(np.arccos(np.clip(abs(cosine), 0, 1)))


    def calculate_posture_features(self, keypoints, bbox, keypoint_scores):
        """综合身体姿态特征计算"""
        features = {}
        # 确保bbox参数正确性
        try:
            img_height = float(bbox[3] - bbox[1])
            img_width = float(bbox[2] - bbox[0])
        except IndexError:
            return None
        
        # 关键点有效性检查
        required_joints = {
            'shoulders': [5, 6],
            'hips': [11, 12],
            'knees': [13, 14],
            'ankles': [15, 16]
        }
        valid_joints = {}
        for part, indices in required_joints.items():
            valid_joints[part] = [
                self._get_valid_point(keypoints[i], bbox[3]) 
                for i in indices 
                if keypoint_scores[i] > self.min_confidence
            ]
            if len(valid_joints[part]) < 1:
                return None  # 关键点不足时返回空

        # 身体比例特征
        shoulders_y = np.mean([p[1] for p in valid_joints['shoulders']])
        hips_y = np.mean([p[1] for p in valid_joints['hips']])
        ankles_y = np.mean([p[1] for p in valid_joints['ankles']])
        
        features['height_ratio'] = (ankles_y - shoulders_y) / (bbox[3]-bbox[1])
        features['head_hip_diff'] = (hips_y - keypoints[0][1]/bbox[3]) if keypoint_scores[0] > self.min_confidence else 0
        
        # 宽高比
        bbox_w = bbox[2]-bbox[0]
        bbox_h = bbox[3]-bbox[1]
        features['aspect_ratio'] = bbox_w / bbox_h if bbox_h > 0 else 0
        
        # 水平姿势检测
        keypoints_y = [keypoints[i][1]/bbox[3] for i in [5,6,11,12,13,14,15,16]]
        features['horizontal_ratio'] = np.std(keypoints_y)/np.ptp(keypoints_y)
        
        # 运动轨迹分析
        centroid = (np.mean([p[0] for p in valid_joints['hips']]), hips_y)
        features['velocity'] = self._calculate_velocity(centroid)
        self.history.append(centroid)
        
        return features

    def is_fall(self, pose_results):
        current_detected = False
        for person in pose_results:
            # 增加数据格式验证
            if not isinstance(person.pred_instances, InstanceData):
                continue
            if not hasattr(person.pred_instances, 'keypoints') or \
            person.pred_instances.keypoints.size == 0:
                continue
            try:
                if not hasattr(person.pred_instances, 'keypoints') or len(person.pred_instances.keypoints) == 0:
                    continue
                    
                keypoints = person.pred_instances.keypoints[0].squeeze()
                keypoint_scores = person.pred_instances.keypoint_scores[0].squeeze()
                bbox = person.pred_instances.bboxes[0].squeeze()
                # 确保img_height是标量值
                img_height = float(bbox[3] - bbox[1])  # 转换为标量高度值
                
                # 数据有效性校验
                if keypoints.shape != (17, 2) or keypoint_scores.shape != (17,):
                    continue
                if (bbox[3]-bbox[1]) < 10:  # 排除过小检测框
                    continue
            except:
                continue

            # 综合特征计算
            features = self.calculate_posture_features(keypoints, bbox, keypoint_scores)
            if not features:
                continue
                
            # 脊柱角度计算
            spine_angle = self.calculate_spine_angle(keypoints, img_height)
            if np.isnan(spine_angle):
                continue

            # 摔倒条件判断
            conditions = {
                'angle': spine_angle > self.angle_threshold,
                'height': features['height_ratio'] < self.height_ratio_threshold,
                'aspect': features['aspect_ratio'] > self.aspect_ratio_threshold,
                'horizontal': features['horizontal_ratio'] < self.horizontal_threshold,
                'motion': features['velocity'] < self.velocity_threshold
            }

            # 动态权重评分系统
            scores = {
                'angle': 0.3 if conditions['angle'] else 0,
                'height': 0.25 if conditions['height'] else 0,
                'aspect': 0.2 if conditions['aspect'] else 0,
                'horizontal': 0.15 if conditions['horizontal'] else 0,
                'motion': 0.1 if conditions['motion'] else 0
            }
            total_score = sum(scores.values())

            # 当且仅当满足以下条件之一时触发：
            # 1. 总分≥0.7且持续3帧
            # 2. 核心三条件同时满足（角度+高度+姿态）
            if (total_score >= 0.7) or (
                conditions['angle'] and 
                conditions['height'] and 
                (conditions['aspect'] or conditions['horizontal'])
            ):
                current_detected = True
                break

        # 改进的连续帧验证
        if current_detected:
            self.consecutive_frames = min(self.consecutive_frames+1, self.min_consecutive_frames*2)
            if self.consecutive_frames >= self.min_consecutive_frames:
                self.consecutive_frames = 0
                return True
        else:
            self.consecutive_frames = max(self.consecutive_frames-2, 0)  # 快速衰减

        return False
