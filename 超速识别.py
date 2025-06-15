import csv
import os
import sys
from datetime import datetime
import json
import cv2
from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QCursor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QFileDialog, QHBoxLayout,
                             QLabel, QMainWindow, QPushButton, QSlider,
                             QSpinBox, QVBoxLayout, QWidget )
from ultralytics import YOLO

from license_plate.detect_plate import license_plate_detection_recognition


def write_speed_record(speed, lpr_str, speed_threshold, csv_file_path):
    """
    将车辆过线速度信息写入CSV文件，包括标题行（如果文件是新的）。

    参数:
    - speed: 过线平均速度，单位km/h。
    - lpr_str: 车牌号码。
    - speed_threshold: 速度阈值，单位km/h。
    - csv_file_path: CSV文件路径。

    CSV格式: 日期, 过线平均速度, 车牌, 速度阈值, 是否超速, 超速百分比
    """
    # 检查文件是否存在，确定是否需要写入标题行
    file_exists = os.path.exists(csv_file_path)

    # 计算是否超速及超速百分比
    is_overspeed = speed > speed_threshold
    overspeed_percentage = ((speed - speed_threshold) / speed_threshold * 100) if is_overspeed else 0

    # 获取当前日期和时间
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 准备写入的数据
    record = [
        current_datetime,
        f"过线平均速度：{round(speed, 2)}km/h",
        f"车牌：{lpr_str}",
        speed_threshold,
        "是" if is_overspeed else "否",
        round(overspeed_percentage, 2)
    ]

    # 写入CSV文件
    with open(csv_file_path, mode='a', newline='', encoding='gb2312') as file:
        writer = csv.writer(file)
        # 如果文件是新创建的，先写入标题行
        if not file_exists:
            writer.writerow(["日期", "过线平均速度", "车牌", "速度阈值", "是否超速", "超速百分比"])
        writer.writerow(record)
        
        
class CarObject:
    """车辆对象类，用于跟踪单个车辆的速度和位置信息"""
    def __init__(self, id, line_1, line_2):
        self.id = int(id)
        self.frame_cnt = 0  # 帧计数器
        self.box_list = []  # 存储车辆边界框的列表
        
        self.line_1 = line_1  # 第一条检测线
        self.line_2 = line_2  # 第二条检测线
        
        self.is_skip_line_1 = False  # 是否已越过第一条线
        self.is_skip_line_2 = False  # 是否已越过第二条线
        
        self.start_frame_cnt = 0  # 开始帧计数
        self.end_frame_cnt = 0  # 结束帧计数
        
        self.start_end_box_list = []  # 用于计算速度的边界框列表
        
        # 像素到实际距离的转换系数 (单位: km/1000像素)
        self.k = 0.01/1000  
        self.fps = 25  # 视频帧率
        self.speed = -1  # 计算得到的速度，初始为-1表示未计算
        
        self.lpr_str = ""  # 车牌识别结果
        
    def update_frame(self):
        """更新帧计数器"""
        self.frame_cnt += 1
    
    def updata_box(self, box, frame):
        """更新车辆边界框信息并计算速度"""
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        
        # 维护最近5个边界框
        self.box_list.append(box)
        self.box_list = self.box_list[-5:]
        
        # 如果两条线都未越过
        if (not self.is_skip_line_1) and (not self.is_skip_line_2):
            if self.is_skip_line(self.box_list, self.line_1):
                self.is_skip_line_1 = True
                self.start_frame_cnt = self.frame_cnt
            
            if self.is_skip_line(self.box_list, self.line_2):
                self.is_skip_line_2 = True
                self.start_frame_cnt = self.frame_cnt
                
        # 如果越过第一条线但未越过第二条线
        if self.is_skip_line_1 and (not self.is_skip_line_2):
            self.start_end_box_list.append(box)
            if not self.lpr_str:  # 如果还未识别车牌
                self.lpr_str, _ = lpr.predict(frame[y1:y2, x1:x2])
            
            if self.is_skip_line(self.box_list, self.line_2):
                self.is_skip_line_2 = True
                self.end_frame_cnt = self.frame_cnt
                
        # 如果越过第二条线但未越过第一条线
        if (not self.is_skip_line_1) and self.is_skip_line_2:
            self.start_end_box_list.append(box)
            if not self.lpr_str:  # 如果还未识别车牌
                self.lpr_str, _ = lpr.predict(frame[y1:y2, x1:x2])
            
            if self.is_skip_line(self.box_list, self.line_1):
                self.is_skip_line_1 = True
                self.end_frame_cnt = self.frame_cnt
                
        # 如果两条线都已越过
        if self.is_skip_line_1 and self.is_skip_line_2:
            if self.speed < 0:  # 如果还未计算速度
                frame_cnt = self.end_frame_cnt - self.start_frame_cnt
                all_time = frame_cnt/self.fps/60/60  # 转换为小时
                line_length = self.calculate_line_length()*self.k  # 计算实际距离
                self.speed = line_length/all_time  # 计算速度(km/h)
                write_speed_record(self.speed, self.lpr_str, threshold, "speed_records.csv")
                
            return True, self.speed
            
        return False, self.speed
    
    def draw_update_box(self, frame, box):
        """绘制车辆边界框和相关信息"""
        is_skip_all, speed = self.updata_box(box, frame)
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 如果已越过两条线，显示速度信息
        if is_skip_all:
            color = (0, 0, 255) if speed > threshold else (0, 255, 0)
            text = f"过线平均速度：{round(speed, 2)}km/h{'，已超速！' if speed > threshold else ''}"
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # 显示车牌信息
        if self.lpr_str:
            color = (0, 0, 255) if speed > threshold else (0, 255, 0)
            cv2.putText(frame, f"车牌：{self.lpr_str}", (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # 绘制车辆轨迹点
        points_list = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in self.box_list]
        for point in points_list:
            color = (0, 0, 255) if speed > threshold else (255, 255, 0)
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, color, -1)
            
        return frame
        
    def calculate_line_length(self):
        """计算车辆行驶轨迹的总长度(像素)"""
        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in self.start_end_box_list]
        total_length = 0
        
        # 计算相邻点之间的距离并累加
        for i in range(len(centers) - 1):
            p1 = centers[i]
            p2 = centers[i + 1]
            distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            total_length += distance
        
        return total_length
    
    def is_skip_line(self, box_list, line):
        """判断车辆是否越过了指定的线"""
        def cross_product(ax, ay, bx, by):
            return ax * by - ay * bx

        def is_intersect(p1, p2, q1, q2):
            r = (p2[0] - p1[0], p2[1] - p1[1])
            s = (q2[0] - q1[0], q2[1] - q1[1])
            qp = (q1[0] - p1[0], q1[1] - p1[1])
            rxs = cross_product(r[0], r[1], s[0], s[1])
            qpxr = cross_product(qp[0], qp[1], r[0], r[1])
            if rxs == 0:
                return False
            t = cross_product(qp[0], qp[1], s[0], s[1]) / rxs
            u = qpxr / rxs
            return 0 <= t <= 1 and 0 <= u <= 1

        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in box_list]
        for i in range(len(centers) - 1):
            for j in range(i + 1, len(centers)):
                if is_intersect(centers[i], centers[j], line[0], line[1]):
                    return True
        return False


class VideoSpeedDetectionApp(QMainWindow):
    """主应用程序窗口"""
    def __init__(self):
        super().__init__()
        
        # 初始化变量
        self.cap = None
        self.model = None
        self.timer = QTimer()
        self.frame = None
        self.car_objects = []
        self.drawing_line = False
        self.current_line = []
        self.line1 = []
        self.line2 = []
        self.is_paused = False
        self.threshold = 15  # 默认超速阈值
        
        # 初始化UI
        self.init_ui()
        
        # 初始化车牌识别模型
        self.lpr = license_plate_detection_recognition()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("车辆测速系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label, 4)
        
        # 控制面板
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, 1)
        
        # 视频控制区域
        video_control_group = QVBoxLayout()
        control_panel.addLayout(video_control_group)
        
        # 视频路径选择
        self.btn_open = QPushButton("选择视频文件")
        self.btn_open.clicked.connect(self.open_video)
        video_control_group.addWidget(self.btn_open)
        
        # 播放/暂停按钮
        self.btn_play_pause = QPushButton("播放")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        video_control_group.addWidget(self.btn_play_pause)
        
        # 检测线设置区域
        line_control_group = QVBoxLayout()
        control_panel.addLayout(line_control_group)
        
        # 检测线1按钮
        self.btn_line1 = QPushButton("设置检测线1")
        self.btn_line1.clicked.connect(lambda: self.start_drawing_line(1))
        line_control_group.addWidget(self.btn_line1)
        
        # 检测线2按钮
        self.btn_line2 = QPushButton("设置检测线2")
        self.btn_line2.clicked.connect(lambda: self.start_drawing_line(2))
        line_control_group.addWidget(self.btn_line2)
        
        # 超速阈值设置
        threshold_group = QHBoxLayout()
        control_panel.addLayout(threshold_group)
        
        threshold_group.addWidget(QLabel("超速阈值(km/h):"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 200)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self.update_threshold)
        threshold_group.addWidget(self.threshold_spin)
        
        # 显示设置区域
        display_group = QVBoxLayout()
        control_panel.addLayout(display_group)
        
        # 显示轨迹复选框
        self.show_track = QCheckBox("显示车辆轨迹")
        self.show_track.setChecked(True)
        display_group.addWidget(self.show_track)
        
        # 显示车牌复选框
        self.show_plate = QCheckBox("显示车牌信息")
        self.show_plate.setChecked(True)
        display_group.addWidget(self.show_plate)
        
        # 显示速度复选框
        self.show_speed = QCheckBox("显示速度信息")
        self.show_speed.setChecked(True)
        display_group.addWidget(self.show_speed)
        
        # 状态栏
        self.statusBar().showMessage("准备就绪")
        
        # 连接定时器
        self.timer.timeout.connect(self.update_frame)
        
        # 尝试自动加载检测线
        self.load_detection_lines()
        self.video_label.setMouseTracking(True)  # 启用鼠标跟踪
        self.setMouseTracking(True)
        
        
    def save_detection_lines(self):
        """自动保存检测线到配置文件"""
        if not (self.line1 and self.line2):
            return
            
        config = {
            'line1': self.line1,
            'line2': self.line2,
            'threshold': self.threshold
        }
        
        with open('detection_lines.json', 'w') as f:
            json.dump(config, f)

    def load_detection_lines(self):
        """自动加载检测线配置"""
        try:
            with open('detection_lines.json', 'r') as f:
                config = json.load(f)
                self.line1 = [tuple(point) for point in config['line1']]
                self.line2 = [tuple(point) for point in config['line2']]
                self.threshold = config['threshold']
                self.threshold_spin.setValue(self.threshold)
                return True
        except:
            return False
        
    def open_video(self):
        """打开视频文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", 
                                                 "Video Files (*.avi *.mp4 *.mov);;All Files (*)", 
                                                 options=options)
        if file_name:
            # 初始化视频捕获
            self.cap = cv2.VideoCapture(file_name)
            
            # 初始化YOLO模型
            self.model_name = 'runs/detect/train/weights/best.pt'
            self.model = YOLO(self.model_name)
            
            # 清空之前的车辆数据
            self.car_objects = []
            
            # 读取第一帧
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.display_frame(frame)
                
                # 启用播放按钮
                self.btn_play_pause.setEnabled(True)
                self.statusBar().showMessage(f"已加载视频: {file_name}")
            else:
                self.statusBar().showMessage("无法读取视频文件")
    
    def toggle_play_pause(self):
        """切换播放/暂停状态"""
        if self.is_paused:
            self.timer.start(30)  # 约30fps
            self.btn_play_pause.setText("暂停")
            self.is_paused = False
        else:
            self.timer.stop()
            self.btn_play_pause.setText("播放")
            self.is_paused = True
    
    def update_threshold(self, value):
        """更新超速阈值"""
        global threshold
        threshold = value
        self.threshold = value
    
    def start_drawing_line(self, line_num):
        """开始绘制检测线"""
        if self.frame is None:
            self.statusBar().showMessage("请先加载视频")
            return
            
        self.drawing_line = True
        self.current_line_num = line_num
        self.current_line = []
        self.statusBar().showMessage(f"请在视频上点击两点绘制检测线{line_num}")







    def display_frame(self, frame):
        """在QLabel上显示视频帧"""
        frame_copy = frame.copy()
        
        # 绘制已设置的检测线
        if self.line1:
            cv2.line(frame_copy, self.line1[0], self.line1[1], (0, 255, 0), 2)
        if self.line2:
            cv2.line(frame_copy, self.line2[0], self.line2[1], (0, 0, 255), 2)
        
        # 如果正在绘制检测线且已经有一个点，显示临时线段
        if self.drawing_line and len(self.current_line) == 1:
            # 获取当前鼠标位置（相对视频帧的坐标）
            mouse_pos = self.get_video_mouse_pos()
            if mouse_pos:
                # 绘制临时线段
                cv2.line(frame_copy, self.current_line[0], mouse_pos, (255, 255, 0), 2)
        
        # 显示处理后的帧
        rgb_image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_video_mouse_pos(self):
        """获取鼠标在视频帧中的坐标（修正版）"""
        if not self.video_label.pixmap() or self.frame is None:
            return None

        # 获取原始视频尺寸
        original_height, original_width = self.frame.shape[:2]
        
        # 获取全局鼠标位置
        global_pos = self.mapFromGlobal(QCursor.pos())
        label_pos = global_pos - self.video_label.pos()
        
        # 计算实际显示区域
        pixmap = self.video_label.pixmap()
        scaled_size = pixmap.size().scaled(
            self.video_label.size(), Qt.KeepAspectRatio)
        
        # 计算显示边距
        x_offset = (self.video_label.width() - scaled_size.width()) // 2
        y_offset = (self.video_label.height() - scaled_size.height()) // 2
        
        # 检查鼠标是否在有效区域
        if not (x_offset <= label_pos.x() < x_offset + scaled_size.width() and
                y_offset <= label_pos.y() < y_offset + scaled_size.height()):
            return None
        
        # 精确坐标转换（使用浮点运算避免误差）
        ratio_x = original_width / scaled_size.width()
        ratio_y = original_height / scaled_size.height()
        
        video_x = int((label_pos.x() - x_offset) * ratio_x)
        video_y = int((label_pos.y() - y_offset) * ratio_y)
        
        # 边界保护
        video_x = max(0, min(video_x, original_width - 1))
        video_y = max(0, min(video_y, original_height - 1))
        
        return (video_x, video_y)


    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        if self.drawing_line and len(self.current_line) == 1:
            # 请求重绘
            self.display_frame(self.frame)
            
    def mousePressEvent(self, event):
        if self.drawing_line and self.video_label.underMouse():
            mouse_pos = self.get_video_mouse_pos()
            if mouse_pos:
                self.current_line.append(mouse_pos)
                if len(self.current_line) == 2:
                    if self.current_line_num == 1:
                        self.line1 = self.current_line.copy()
                    else:
                        self.line2 = self.current_line.copy()
                    self.drawing_line = False
                    self.save_detection_lines()
                    self.display_frame(self.frame)


    
                     
    
    def update_frame(self):
        """更新视频帧"""
        if self.cap and self.model:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                
                # 运行YOLOv5跟踪
                if self.model_name == 'yolov5su.pt':
                    results = self.model.track(frame, persist=True, classes=[2, 5, 7])  # 2:汽车, 5:公交车, 7:卡车
                else:
                    results = self.model.track(frame, persist=True, classes=[3, 4, 5, 8])
                
                
                # 绘制检测线
                if self.line1:
                    cv2.line(frame, self.line1[0], self.line1[1], (0, 255, 0), 2)
                if self.line2:
                    cv2.line(frame, self.line2[0], self.line2[1], (0, 0, 255), 2)
                
                # 获取检测结果
                id_list = []
                xyxy_list = []
                if len(results) > 0 and results[0].boxes.id is not None:
                    id_list = results[0].boxes.id.cpu().numpy().tolist()
                    xyxy_list = results[0].boxes.xyxy.cpu().numpy().tolist()
                
                # 处理每个检测到的车辆
                for id, xyxy in zip(id_list, xyxy_list):
                    existing_obj = next((obj for obj in self.car_objects if obj.id == id), None)
                    
                    if existing_obj:
                        # 更新现有车辆对象
                        existing_obj.draw_update_box(frame, xyxy)
                    else:
                        # 创建新车辆对象
                        if self.line1 and self.line2:  # 确保两条检测线都已设置
                            new_obj = CarObject(id, self.line1, self.line2)
                            new_obj.draw_update_box(frame, xyxy)
                            self.car_objects.append(new_obj)
                
                # 更新每个车辆对象的帧计数器
                for obj in self.car_objects:
                    obj.update_frame()
                
                # 显示处理后的帧
                self.display_frame(frame)
            else:
                # 视频结束
                self.timer.stop()
                self.btn_play_pause.setText("播放")
                self.is_paused = True
                self.statusBar().showMessage("视频播放结束")
    
 
    
    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        if self.cap:
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


if __name__ == '__main__':
    # 初始化车牌识别模型
    lpr = license_plate_detection_recognition()
    
    # 全局超速阈值
    threshold = 15
    
    # 创建应用
    app = QApplication(sys.argv)
    window = VideoSpeedDetectionApp()
    window.show()
    sys.exit(app.exec_())