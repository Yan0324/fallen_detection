import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                            QPushButton, QHBoxLayout, QVBoxLayout, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from styles import MAIN_STYLE, STATUS_LED_STYLE, STAT_ITEM_STYLE, APP_FONT
# 新增导入
from PyQt5.QtGui import QImage, QPixmap
from video_thread import VideoProcessor

class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人物颠倒监测系统")
        self.setMinimumSize(1280, 720)
        self.setWindowIcon(QIcon("icon.png"))
        self.init_ui()
        self.apply_styles()

        self.video_processor = None  # 新增视频处理器
        self.alarm_counter = 0  # 新增报警计数器

    def init_ui(self):
        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 视频区域
        main_layout.addWidget(self.create_video_section(), 3)
        
        # 控制面板
        main_layout.addWidget(self.create_control_panel(), 1)

    def create_video_section(self):
        frame = QFrame()
        frame.setObjectName("videoFrame")
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题栏
        title_bar = QWidget()
        title_bar.setFixedHeight(40)
        hbox = QHBoxLayout(title_bar)
        hbox.setContentsMargins(15, 0, 15, 0)
        
        self.camera_label = QLabel("摄像头 1 - 实时画面")
        self.status_led = QLabel("●")
        hbox.addWidget(self.camera_label)
        hbox.addWidget(self.status_led)
        
        # 视频显示区
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(800, 600)
        self.video_display.setStyleSheet("background: #2D2D2D; border-radius: 10px;")
        self.video_display.setText("视频流显示区域")
        
        layout.addWidget(title_bar)
        layout.addWidget(self.video_display)
        
        return frame

    def create_control_panel(self):
        panel = QFrame()
        panel.setObjectName("controlPanel")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 控制组件
        layout.addWidget(QLabel("控制面板", alignment=Qt.AlignCenter))
        # 创建按钮时需要保存为实例变量
        self.start_btn = self.create_button("启动监测")  # 增加self.
        self.pause_btn = self.create_button("暂停监测")  # 增加self.

        self.start_btn.clicked.connect(self.start_monitoring)
        self.pause_btn.clicked.connect(self.stop_monitoring)

    
        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addSpacing(20)
        layout.addWidget(self.create_stats_card())
        layout.addStretch()
        layout.addWidget(self.create_button("系统设置"))
        layout.addWidget(self.create_button("导出记录"))
        
        return panel

    def create_button(self, text):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def create_stats_card(self):
        card = QFrame()
        card.setObjectName("statsCard")
        
        vbox = QVBoxLayout(card)
        vbox.addWidget(QLabel("今日统计"))
        vbox.addWidget(StatItem("总检测次数", "0"))
        vbox.addWidget(StatItem("检测人数", "0"))
        vbox.addWidget(StatItem("异常事件", "0"))
        vbox.addWidget(StatItem("识别准确率", "98.2%"))
        
        return card

    def apply_styles(self):
        self.setStyleSheet(MAIN_STYLE)
        self.status_led.setStyleSheet(STATUS_LED_STYLE)
        self.setFont(APP_FONT)
    # 新增视频更新槽函数
    def update_video_frame(self, image):
        self.video_display.setPixmap(QPixmap.fromImage(image))

    # 新增监控控制方法
    def start_monitoring(self):
        if not self.video_processor:
            self.video_processor = VideoProcessor(
                # config_path='../../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py',
                # checkpoint_path='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth'
                pose_config='../../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py',
                pose_checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth',
                det_config='../../mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', 
                det_checkpoint='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
            )
            self.video_processor.frame_processed.connect(self.update_video_frame)
            self.video_processor.fall_detected.connect(self.emit_alarm)  # 连接信号
        self.video_processor.start()
        self.status_led.setStyleSheet("color: #00FF00;")

    def stop_monitoring(self):
        if self.video_processor:
            self.video_processor.stop()
            self.video_processor.quit()
            self.video_processor.wait()
            self.video_processor = None
            # 确保视频处理器线程已经停止
            # 恢复视频显示区域为初始状态
        self.video_display.setPixmap(QPixmap())  # 清除当前显示的视频帧
        self.video_display.setText("视频流显示区域")  # 恢复提示文字
        self.video_display.setStyleSheet("background: #2D2D2D; border-radius: 10px;")  # 恢复背景样式
        self.status_led.setStyleSheet("color: #FF0000;")

    def emit_alarm(self):
        # 更新UI显示
        self.status_led.setStyleSheet("color: #FF0000;")
        self.camera_label.setText("警报！检测到摔倒！")
        
        # 更新统计信息
        self.alarm_counter += 1
        stats = self.findChild(QLabel, "stat_异常事件")
        if stats:
            stats.value_label.setText(str(self.alarm_counter))

class StatItem(QLabel):
    def __init__(self, title, value):
        super().__init__()
        self.setObjectName(f"stat_{title}")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        title_label = QLabel(f"{title}:")
        title_label.setStyleSheet(STAT_ITEM_STYLE["title"])
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(STAT_ITEM_STYLE["value"])
        
        layout.addWidget(title_label)
        layout.addStretch()
        layout.addWidget(self.value_label)

if __name__ == "__main__":
    import os
    # 过滤Qt调试信息
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    window = ModernWindow()
    window.show()
    sys.exit(app.exec_())