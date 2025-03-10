import numpy as np

class FallDetection:
    def __init__(self, 
                 angle_threshold=40,
                 height_ratio_threshold=0.4,
                 min_confidence=0.3,
                 head_hip_threshold=0.3,
                 aspect_ratio_threshold=1.5,
                 horizontal_threshold=0.15,
                 min_consecutive_frames=2):
        """优化参数和新增水平姿势检测"""
        self.angle_threshold = angle_threshold
        self.height_ratio_threshold = height_ratio_threshold
        self.min_confidence = min_confidence
        self.head_hip_threshold = head_hip_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.horizontal_threshold = horizontal_threshold
        self.min_consecutive_frames = min_consecutive_frames
        
        self.consecutive_frames = 0
        self.last_detection_params = {}

    def get_valid_point(self, point, default=0):
        """处理无效坐标点"""
        if np.isnan(point[0]) or np.isnan(point[1]):
            return (default, default)
        return (point[0], point[1])

    def calculate_spine_angle(self, keypoints):
        """计算脊柱与垂直方向的夹角"""
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
        
        if np.linalg.norm(spine_vector) < 1e-6:
            return 0
            
        cosine = np.dot(spine_vector, vertical_vector) / (
            np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
        return np.degrees(np.arccos(np.clip(abs(cosine), 0, 1)))

    def calculate_horizontal_ratio(self, keypoints):
        """计算身体水平姿势比例"""
        # 使用肩、髋、膝、踝的水平差异
        shoulders = [keypoints[5], keypoints[6]]
        hips = [keypoints[11], keypoints[12]]
        knees = [keypoints[13], keypoints[14]]
        ankles = [keypoints[15], keypoints[16]]
        
        # 收集有效的y坐标
        y_coords = []
        for joint in shoulders + hips + knees + ankles:
            if not np.isnan(joint[1]):
                y_coords.append(joint[1])
        
        if len(y_coords) < 4:
            return 1.0  # 默认非水平状态
        
        # 计算垂直方向的离散程度
        y_range = max(y_coords) - min(y_coords)
        if y_range < 1e-6:
            return 0.0
        std_dev = np.std(y_coords)
        return std_dev / y_range

    def is_fall(self, pose_results):
        current_detected = False
        for person in pose_results:
            try:
                if not hasattr(person.pred_instances, 'keypoints') or \
                len(person.pred_instances.keypoints) == 0:
                    continue
                    
                keypoints = person.pred_instances.keypoints[0].squeeze()
                keypoint_scores = person.pred_instances.keypoint_scores[0].squeeze()
                bbox = person.pred_instances.bboxes[0].squeeze()
                
                if keypoints.ndim != 2 or keypoints.shape != (17, 2):
                    continue
                if keypoint_scores.ndim != 1 or keypoint_scores.shape != (17,):
                    continue
            except Exception as e:
                print(f"数据解析错误: {str(e)}")
                continue

            # 关键点有效性验证
            required_joints = [5, 6, 11, 12, 13, 14, 15, 16]  # 增加下肢关键点
            if any(keypoint_scores[i] < self.min_confidence for i in required_joints):
                continue

            try:
                img_height = abs(bbox[3] - bbox[1])
                if img_height < 10:
                    continue

                # 核心参数计算
                spine_angle = self.calculate_spine_angle(keypoints)
                horizontal_ratio = self.calculate_horizontal_ratio(keypoints)
                
                # 头部位置计算
                head_coords = []
                if keypoint_scores[0] > self.min_confidence:
                    head_coords.append(float(keypoints[0][1]))
                for i in [3,4]:
                    if keypoint_scores[i] > self.min_confidence:
                        head_coords.append(float(keypoints[i][1]))
                head_y = np.mean(head_coords) if head_coords else np.nan
                
                # 髋部位置
                mid_hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
                
                # 宽高比
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
                
                # 身高比例
                ankles = [keypoints[i] for i in [15,16] if keypoint_scores[i] > self.min_confidence]
                if not ankles:
                    continue
                max_ankle_y = max(a[1] for a in ankles)
                shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
                body_height = abs(shoulder_y - max_ankle_y)
                height_ratio = body_height / img_height
            except Exception as e:
                print(f"参数计算错误: {str(e)}")
                continue

            # 调试信息
            debug_info = {
                'spine_angle': spine_angle,
                'head_hip_diff': abs(head_y - mid_hip_y) if not np.isnan(head_y) else 0,
                'aspect_ratio': aspect_ratio,
                'height_ratio': height_ratio,
                'horizontal_ratio': horizontal_ratio  # 新增调试信息
            }
            print("当前检测参数:", {k: round(v,2) for k,v in debug_info.items()})

            # 多条件判断
            angle_cond = spine_angle > self.angle_threshold
            head_cond = (head_y - mid_hip_y) > (img_height * self.head_hip_threshold)
            aspect_cond = aspect_ratio > self.aspect_ratio_threshold
            height_cond = height_ratio < self.height_ratio_threshold
            horizontal_cond = horizontal_ratio < self.horizontal_threshold

            # 条件组合逻辑优化
            conditions = [angle_cond, head_cond, aspect_cond, height_cond, horizontal_cond]
            
            # 触发条件：满足任意两个条件，或水平姿势+任一其他条件
            condition_count = sum(conditions)
            horizontal_plus_other = horizontal_cond and condition_count >= 1
            
            if condition_count >= 2 or horizontal_plus_other:
                current_detected = True
                break

        # 连续帧判断
        if current_detected:
            self.consecutive_frames += 1
            if self.consecutive_frames >= self.min_consecutive_frames:
                print(f"触发摔倒检测！连续帧数：{self.consecutive_frames}")
                self.consecutive_frames = 0
                return True
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

        return False
