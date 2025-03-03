from PyQt5.QtGui import QFont

# 主窗口样式
MAIN_STYLE = """
QMainWindow {
    background: #1A1A1A;
}

/* 视频区域样式 */
#videoFrame, #controlPanel {
    background: #252526;
    border-radius: 12px;
    border: 1px solid #383838;
}

/* 统计卡片样式 */
QLabel#statsCard {
    background: #2D2D30;
    border-radius: 8px;
    padding: 20px 15px;
}

/* 通用标签样式 */
QLabel {
    color: #D4D4D4;
    font: 14px 'Segoe UI';
}

/* 按钮基础样式 */
QPushButton {
    color: #FFFFFF;
    background: #3E3E42;
    border: none;
    padding: 12px;
    border-radius: 8px;
    font: bold 14px;
}

QPushButton:hover { 
    background: #505050; 
}

QPushButton:pressed { 
    background: #007ACC; 
}

/* 统计项特定样式 */
QLabel[objectName^="stat_"] {
    border-radius: 6px;
    padding: 15px;
    min-height: 40px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2D2D30, stop:1 #252526);
}
"""

# 状态指示灯样式
STATUS_LED_STYLE = """
QLabel {
    color: #808080;
    font: 24px;
}
"""

# 统计项字体样式
STAT_ITEM_STYLE = {
    "title": "color: #858585; font: bold 14px;",
    "value": "color: #D4D4D4; font: bold 16px;"
}

# 全局字体设置
APP_FONT = QFont("Segoe UI", 10)

# 在MAIN_STYLE中增加视频标签样式
MAIN_STYLE += """
#video_display {
    border-radius: 10px;
    min-width: 800px;
    min-height: 600px;
}
"""


