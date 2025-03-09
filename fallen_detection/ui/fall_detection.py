import numpy as np

class FallDetection:
    def __init__(self, angle_threshold=60, height_ratio_threshold=0.4,
                 min_confidence=0.3):
        self.angle_threshold = angle_threshold
        self.height_ratio_threshold = height_ratio_threshold
        self.min_confidence = min_confidence

    def get_valid_point(self, point, default=0):
        """处理无效坐标点"""
        if np.isnan(point[0]) or np.isnan(point[1]):
            return (default, default)
        return (point[0], point[1])

    def calculate_spine_angle(self, keypoints):
        """修正后的脊柱角度计算方法"""
        # COCO关键点索引：
        # 11:左髋，12:右髋，5:左肩，6:右肩
        left_hip = self.get_valid_point(keypoints[11])
        right_hip = self.get_valid_point(keypoints[12])
        left_shoulder = self.get_valid_point(keypoints[5])
        right_shoulder = self.get_valid_point(keypoints[6])

        mid_hip = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        )
        
        mid_shoulder = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        )

        spine_vector = np.array([mid_shoulder[0] - mid_hip[0], 
                               mid_shoulder[1] - mid_hip[1]])
        vertical_vector = np.array([0, 1])
        
        # 计算角度并避免零向量
        if np.linalg.norm(spine_vector) < 1e-6:
            return 0
            
        cosine = np.dot(spine_vector, vertical_vector) / (
            np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
        return np.degrees(np.arccos(np.clip(abs(cosine), 0, 1)))

    def is_fall(self, pose_results):
        for person in pose_results:
            # 确保存在必要的属性
            if not hasattr(person.pred_instances, 'keypoints') or \
            not hasattr(person.pred_instances, 'keypoint_scores') or \
            not hasattr(person.pred_instances, 'bboxes'):
                continue

            # 检查数组维度
            if len(person.pred_instances.keypoints) == 0 or \
            len(person.pred_instances.keypoint_scores) == 0 or \
            len(person.pred_instances.bboxes) == 0:
                continue

            # 获取第一个检测实例的数据（假设单个人检测）
            try:
                keypoints = person.pred_instances.keypoints[0]  # shape: (17,2)
                keypoint_scores = person.pred_instances.keypoint_scores[0]  # shape: (17,)
                bbox = person.pred_instances.bboxes[0]         # shape: (4,)
            except IndexError:
                continue

            # 验证数据维度
            if keypoints.shape != (17, 2) or keypoint_scores.shape != (17,):
                continue

            # 有效关键点过滤
            valid_points = sum(score > self.min_confidence for score in keypoint_scores)
            if valid_points < 6:
                continue

            # 计算脊柱角度
            spine_angle = self.calculate_spine_angle(keypoints)

            # 计算身高比例
            img_height = abs(bbox[3] - bbox[1])
            if img_height < 1e-6:  # 防止除零错误
                continue

            # 获取脚踝位置（当关键点可见时）
            ankle_indices = [15, 16]
            valid_ankles = [
                keypoints[i] for i in ankle_indices 
                if keypoint_scores[i] > self.min_confidence
            ]
            if not valid_ankles:
                continue

            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            max_ankle_y = max(ankle[1] for ankle in valid_ankles)
            body_height = abs(shoulder_y - max_ankle_y)
            height_ratio = body_height / img_height

            # 综合判断条件
            if (spine_angle > self.angle_threshold and 
                height_ratio < self.height_ratio_threshold):
                return True
        print("没有检测到摔倒")    
        return False

