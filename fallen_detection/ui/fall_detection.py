import numpy as np
from collections import deque

class FallDetection:
    def __init__(self):
        # 使用COCO-17关键点正确定义
        self.KEYPOINT_IDS = {
            'left_hip': 11,
            'right_hip': 12,
            'left_shoulder': 5,
            'right_shoulder': 2  # 根据实际模型可能需要调整
        }
        
        # 动态阈值参数（保持不变）
        self.angle_threshold = 60
        self.height_ratio_threshold = 0.4
        self.velocity_threshold = 0.3
        self.consecutive_frames = 5
        
        # 状态跟踪（保持不变）
        self.reference_height = None
        self.height_history = deque(maxlen=30)
        self.fall_history = deque(maxlen=self.consecutive_frames)

    def validate_keypoints(self, keypoints):
        """验证关键点有效性"""
        return (keypoints.ndim == 2 and 
                keypoints.shape[0] >= 17 and  # COCO要求至少17个关键点
                keypoints.shape[1] >= 3)      # 包含(x,y,score)

    def calculate_spine_angle(self, keypoints):
        """修复后的脊柱角度计算"""
        if not self.validate_keypoints(keypoints):
            return None

        try:
            # 使用安全访问方式
            kp = self.KEYPOINT_IDS
            left_hip = keypoints[kp['left_hip']][:2]
            right_hip = keypoints[kp['right_hip']][:2]
            left_shoulder = keypoints[kp['left_shoulder']][:2]
            right_shoulder = keypoints[kp['right_shoulder']][:2]

            # 检查关键点置信度
            min_confidence = 0.3
            if (keypoints[kp['left_hip']][2] < min_confidence or
                keypoints[kp['right_hip']][2] < min_confidence or
                keypoints[kp['left_shoulder']][2] < min_confidence or
                keypoints[kp['right_shoulder']][2] < min_confidence):
                return None

            # 计算中点
            mid_hip = [(left_hip[0] + right_hip[0])/2, 
                      (left_hip[1] + right_hip[1])/2]
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2,
                           (left_shoulder[1] + right_shoulder[1])/2]
            
            # 计算向量（Y轴方向修正）
            spine_vector = np.array([mid_shoulder[0] - mid_hip[0],
                                   mid_hip[1] - mid_shoulder[1]])
            
            # 处理除以零情况
            if spine_vector[1] == 0:
                return 90.0  # 水平状态
            
            angle = np.degrees(np.arctan2(abs(spine_vector[0]), spine_vector[1]))
            return angle
        except (IndexError, KeyError) as e:
            print(f"关键点计算错误: {str(e)}")
            return None

    def update_reference_height(self, current_height):
        """动态更新参考身高"""
        if self.reference_height is None:
            self.reference_height = current_height
        else:
            # 指数平滑更新
            self.reference_height = 0.9*self.reference_height + 0.1*current_height

    def calculate_velocity(self):
        """计算高度变化速度"""
        if len(self.height_history) < 2:
            return 0
        return (self.height_history[-2] - self.height_history[-1])  # 下降速度为正值

    def is_fall(self, pose_results):
        """改进后的摔倒判断逻辑"""
        # 每次检测前重置当前参数
        self.current_angle = 0.0
        self.current_height_ratio = 1.0
        self.current_velocity = 0.0
        fall_condition = False

        if not pose_results:
            return False

        # 初始化最佳候选参数
        max_priority = -float('inf')
        best_candidate = {
            'angle': 0,
            'ratio': 1.0,
            'velocity': 0
        }

        for person in pose_results:
            try:
                keypoints = person.pred_instances.keypoints.squeeze()
                # 过滤低置信度关键点（髋部和肩部置信度需＞0.3）
                if (keypoints.shape[0] < 13 or 
                    keypoints[11][2] < 0.3 or  # 左髋置信度
                    keypoints[8][2] < 0.3 or   # 右髋置信度
                    keypoints[5][2] < 0.3 or   # 左肩置信度
                    keypoints[2][2] < 0.3):    # 右肩置信度
                    continue

                # 1. 计算当前身高（颈椎到髋部的距离）
                mid_shoulder_y = (keypoints[5][1] + keypoints[2][1])/2
                mid_hip_y = (keypoints[11][1] + keypoints[8][1])/2
                current_height = abs(mid_shoulder_y - mid_hip_y)

                # 2. 更新参考身高（动态校准）
                if len(self.height_history) < 30:
                    self.height_history.append(current_height)
                self.update_reference_height(np.median(self.height_history))

                # 3. 计算各项指标
                spine_angle = self.calculate_spine_angle(keypoints) or 0.0
                height_ratio = current_height / self.reference_height if self.reference_height else 1.0
                velocity = self.calculate_velocity()

                # 4. 计算优先级（用于多目标中选择最可能摔倒的）
                priority = (
                    spine_angle * 0.5 + 
                    (1 - height_ratio) * 0.3 + 
                    velocity * 0.2
                )

                # 更新最佳候选参数
                if priority > max_priority:
                    max_priority = priority
                    best_candidate = {
                        'angle': spine_angle,
                        'ratio': height_ratio,
                        'velocity': velocity
                    }

                # 5. 多条件联合判断（需要连续满足条件）
                current_condition = (
                    spine_angle > self.angle_threshold and
                    height_ratio < self.height_ratio_threshold and
                    velocity > self.velocity_threshold
                )
                self.fall_history.append(current_condition)

            except Exception as e:
                print(f"Error processing pose: {e}")
                continue

        # 更新最终检测参数（取最佳候选）
        self.current_angle = best_candidate['angle']
        self.current_height_ratio = best_candidate['ratio']
        self.current_velocity = best_candidate['velocity']

        # 检查连续帧条件
        if len(self.fall_history) == self.consecutive_frames:
            fall_condition = sum(self.fall_history) >= self.consecutive_frames - 1

        return fall_condition

