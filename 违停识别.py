import pkg_resources
try:
    pkg_resources.get_distribution('opencv-python-rolling')
    print("opencv-python-rolling 已安装")
except pkg_resources.DistributionNotFound:
    print("opencv-python-rolling 未安装")
    print("pip install opencv-python-rolling")

import sys
import os
import json
import cv2
import numpy as np
import torch
from datetime import datetime, timedelta
from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QSpinBox,
                             QCheckBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

from license_plate.detect_plate import license_plate_detection_recognition
 

class ParkingSystem:
    def __init__(self):
        
        self.parking_records = {}
    
    def add_vehicle(self, track_id, plate):
        if track_id not in self.parking_records:
            self.parking_records[track_id] = {
                'start_time': datetime.now(),
                'plate': plate,
                'last_seen': datetime.now()
            }
        else:
            self.parking_records[track_id]['last_seen'] = datetime.now()
    
 
    
    def get_parking_time(self, track_id):
        if track_id not in self.parking_records:
            return "0分钟"
        duration = datetime.now() - self.parking_records[track_id]['start_time']
        return f"{int(duration.total_seconds() / 60)}分钟"
    
    def cleanup(self, max_inactive_minutes=5):
        current_time = datetime.now()
        for track_id in list(self.parking_records.keys()):
            inactive_time = current_time - self.parking_records[track_id]['last_seen']
            if inactive_time.total_seconds() > max_inactive_minutes * 60:
                del self.parking_records[track_id]

class ROIEditor:
    def clear_rois(self):
        self.rois = []
        self.save_rois()

    def __init__(self):
        self.rois = []
        self.current_roi = None
        self.drawing = False
        self.roi_file = "roi_config.json"
        self.load_rois()
    
    def start_drawing(self, pos):
        self.current_roi = {'points': [pos]}
        self.drawing = True
    
    def add_point(self, pos):
        if self.drawing and self.current_roi:
            self.current_roi['points'].append(pos)
    
    def finish_drawing(self):
        if self.drawing and self.current_roi and len(self.current_roi['points']) > 2:
            self.rois.append(self.current_roi)
            self.save_rois()
        self.current_roi = None
        self.drawing = False
    
    def is_point_in_roi(self, point):#识别判定点是否在中间
        for roi in self.rois:
            polygon = np.array([[p.x(), p.y()] for p in roi['points']], dtype=np.int32)
            if cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0:
                return True
        return False
    
    def load_rois(self):
        if os.path.exists(self.roi_file):
            try:
                with open(self.roi_file, 'r') as f:
                    self.rois = [{'points': [QPoint(p[0], p[1]) for p in roi['points']]} 
                               for roi in json.load(f)]
            except:
                pass
    
    def save_rois(self):
        with open(self.roi_file, 'w') as f:
            json.dump([{'points': [[p.x(), p.y()] for p in roi['points']]} 
                      for roi in self.rois], f, indent=2)

class VehicleTracker:
    def __init__(self):
        self.model = YOLO('runs/detect/train/weights/best.pt')
        self.lpr = license_plate_detection_recognition()
        
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )
        
        self.track_history = {}
        self.target_classes = ['car', 'truck', 'bus', "van"]
        self.class_names = self.model.names
    
    def process_frame(self, frame, roi_editor, skip_frames=0):
        results = self.model(frame)
        bboxes = []
        
        for result in results:
            for r in result.boxes.data.tolist():
                
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
            
                class_name = self.class_names.get(class_id)
                
                if class_name in self.target_classes and score > 0.25:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if roi_editor.is_point_in_roi(center):
                        plate_img = frame[y1:y2, x1:x2]
                        lpr_str = self.lpr.predict2(plate_img)
                        print(lpr_str)
                        if len(lpr_str) < 3 :continue
                        bboxes.append((x1, y1, x2, y2, class_id, score, lpr_str))
        
        xywhs = torch.Tensor([[int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1] 
                             for x1, y1, x2, y2, *_ in bboxes])
        confss = torch.Tensor([conf for *_, conf, _ in bboxes])
        clss = [cls_id for *_, cls_id, _, _ in bboxes]
        lpr_list = [lpr_str for *_, lpr_str in bboxes]
        
        outputs = self.deepsort.update(xywhs, confss, clss, frame, lpr_list)
        
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = map(int, value[:6])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
                
        out_list = []
        for value in outputs:
            x1, y1, x2, y2, cls_, track_id = map(int, value[:6])
            out_list.append((x1, y1, x2, y2, cls_, track_id, value[6]))
        
        
        return out_list

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("违停识别系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化组件
        self.tracker = VehicleTracker()
        self.roi_editor = ROIEditor()
        self.parking_system = ParkingSystem()
        
        # 视频相关
        self.cap = None
        self.out = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.is_playing = False
        self.show_trajectory = True
        self.show_plate = True
        self.show_parking_info = True
        
        # 当前帧图像
        self.current_frame = None
        self.current_pixmap = None
        
        # UI布局
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.mousePressEvent = self.handle_mouse_click
        main_layout.addWidget(self.video_label, 70)
        
        # 控制面板
        control_panel = QVBoxLayout()
        
        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.open_btn = QPushButton("打开视频")
        self.open_btn.clicked.connect(self.open_video)
        file_layout.addWidget(self.open_btn)
        
        self.save_btn = QPushButton("保存视频")
        self.save_btn.clicked.connect(self.save_video)
        file_layout.addWidget(self.save_btn)
        file_group.setLayout(file_layout)
        control_panel.addWidget(file_group)
        
        # ROI绘制
        roi_group = QGroupBox("ROI绘制")
        roi_layout = QVBoxLayout()
        self.draw_btn = QPushButton("开始绘制")
        self.draw_btn.clicked.connect(self.toggle_drawing)
        roi_layout.addWidget(self.draw_btn)
        
        self.finish_btn = QPushButton("完成绘制")
        self.finish_btn.clicked.connect(self.finish_drawing)
        roi_layout.addWidget(self.finish_btn)
        
        self.clear_btn = QPushButton("清除标线")
        self.clear_btn.clicked.connect(self.roi_editor.clear_rois)
        roi_layout.addWidget(self.clear_btn)
        roi_group.setLayout(roi_layout)
        control_panel.addWidget(roi_group)
        
        # 播放控制
        play_group = QGroupBox("播放控制")
        play_layout = QVBoxLayout()
        self.play_btn = QPushButton("播放/暂停")
        self.play_btn.clicked.connect(self.toggle_play)
        play_layout.addWidget(self.play_btn)
        
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 30)
        self.skip_spin.setPrefix("跳帧数: ")
        play_layout.addWidget(self.skip_spin)
        play_group.setLayout(play_layout)
        control_panel.addWidget(play_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        self.trajectory_cb = QCheckBox("显示轨迹")
        self.trajectory_cb.setChecked(True)
        display_layout.addWidget(self.trajectory_cb)
        
        self.plate_cb = QCheckBox("显示车牌")
        self.plate_cb.setChecked(True)
        display_layout.addWidget(self.plate_cb)
        
        self.info_cb = QCheckBox("显示违停信息")
        self.info_cb.setChecked(True)
        display_layout.addWidget(self.info_cb)
        display_group.setLayout(display_layout)
        control_panel.addWidget(display_group)
        
        # 停车信息
        self.info_label = QLabel("违章停车信息将显示在这里")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("background-color: white; padding: 5px;")
        control_panel.addWidget(self.info_label)
        
        control_panel.addStretch()
        
        # 设置主布局
        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        main_layout.addWidget(control_widget, 30)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.update_display()
                self.frame_count = 0
                self.is_playing = False
                self.timer.stop()
    
    def save_video(self):
        if not self.cap:
            QMessageBox.warning(self, "警告", "请先打开视频文件!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "保存视频", "", "MP4文件 (*.mp4)")
        if file_path:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            QMessageBox.information(self, "提示", "视频保存已开始，请播放视频")
    
    def toggle_play(self):
        if not self.cap:
            return
        self.is_playing = not self.is_playing
        self.timer.start(30) if self.is_playing else self.timer.stop()
    
    def toggle_drawing(self):
        if not self.current_pixmap:
            QMessageBox.warning(self, "警告", "请先打开视频文件!")
            return
        self.roi_editor.drawing = not self.roi_editor.drawing
        self.draw_btn.setText("取消绘制" if self.roi_editor.drawing else "开始绘制")
        self.update_display()
    
    def finish_drawing(self):
        self.roi_editor.finish_drawing()
        self.roi_editor.drawing = False
        self.draw_btn.setText("开始绘制")
        self.update_display()
    
    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        
        for _ in range(self.skip_spin.value() + 1):
            ret, frame = self.cap.read()
            self.frame_count += 1
            if not ret:
                self.timer.stop()
                self.is_playing = False
                if self.out:
                    self.out.release()
                    self.out = None
                    QMessageBox.information(self, "提示", "视频保存完成")
                return
        
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.process_frame()
        
        if self.out and self.current_pixmap:
            # 将QPixmap转换回OpenCV格式
            qimg = self.current_pixmap.toImage()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)[:, :, :3]
            self.out.write(arr)
    
    def process_frame(self):
        if self.current_frame is None:
            return
            
        # 创建QPixmap
        height, width = self.current_frame.shape[:2]
        q_img = QImage(self.current_frame.data, width, height, 3*width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 绘制ROI和检测结果
        painter = QPainter(pixmap)
        self.draw_rois(painter)
        
        if not self.roi_editor.drawing:
            bboxes = self.tracker.process_frame(self.current_frame, self.roi_editor, self.skip_spin.value())
            parking_info = []
            
            for x1, y1, x2, y2, cls_id, track_id, lpr_str in bboxes:
                # 随机颜色
                color = QColor((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2-x1, y2-y1)
                
                # 轨迹
                if self.trajectory_cb.isChecked() and track_id in self.tracker.track_history:
                    history = self.tracker.track_history[track_id]
                    for i in range(1, len(history)):
                        painter.drawLine(history[i-1][0], history[i-1][1], history[i][0], history[i][1])
                
                # 标签
                label = f"ID:{track_id}"
                if self.plate_cb.isChecked():
                    label += f" {lpr_str}"
                
                text_rect = painter.fontMetrics().boundingRect(label)
                painter.fillRect(x1, y1-text_rect.height()-4, text_rect.width()+4, text_rect.height()+4, color)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(x1+2, y1-4, label)
                
                # 停车信息
                self.parking_system.add_vehicle(track_id, lpr_str)
                if self.info_cb.isChecked():
                    parking_info.append(
                        f"违停车辆ID:{track_id} 车牌:{lpr_str}\n"
                        f"停车时间:{self.parking_system.get_parking_time(track_id)} "
                        "------------------"
                    )
            
            if parking_info:
                self.info_label.setText("\n".join(parking_info))
            
            self.parking_system.cleanup()
        
        painter.end()
        self.current_pixmap = pixmap
        self.update_display()
    
    def draw_rois(self, painter):
        # 绘制ROI
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        
        for roi in self.roi_editor.rois:
            points = roi['points']
            for i in range(len(points)-1):
                painter.drawLine(points[i], points[i+1])
            if len(points) > 2:
                painter.drawLine(points[-1], points[0])
        
        if self.roi_editor.drawing and self.roi_editor.current_roi:
            pen.setColor(QColor(255, 255, 0))
            painter.setPen(pen)
            for i, point in enumerate(self.roi_editor.current_roi['points']):
                painter.drawEllipse(point, 5, 5)
                if i > 0:
                    painter.drawLine(self.roi_editor.current_roi['points'][i-1], point)
    
    def update_display(self):
        if self.current_pixmap:
            self.video_label.setPixmap(self.current_pixmap.scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
    
    def handle_mouse_click(self, event):
        if not self.current_pixmap or not self.roi_editor.drawing:
            return
        
        # 获取视频标签的尺寸和位置
        label_size = self.video_label.size()
        pixmap_size = self.current_pixmap.size()
        
        # 计算缩放比例和偏移
        scale = min(label_size.width()/pixmap_size.width(), 
                   label_size.height()/pixmap_size.height())
        offset_x = (label_size.width() - pixmap_size.width()*scale) / 2
        offset_y = (label_size.height() - pixmap_size.height()*scale) / 2
        
        # 转换为原始图像坐标
        img_x = int((event.pos().x() - offset_x) / scale)
        img_y = int((event.pos().y() - offset_y) / scale)
        
        # 确保坐标在图像范围内
        img_x = max(0, min(img_x, pixmap_size.width()-1))
        img_y = max(0, min(img_y, pixmap_size.height()-1))
        
        if self.roi_editor.drawing:
            if not self.roi_editor.current_roi:
                self.roi_editor.start_drawing(QPoint(img_x, img_y))
            else:
                self.roi_editor.add_point(QPoint(img_x, img_y))
            
            # 只更新绘图，不处理视频帧
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
                q_img = QImage(self.current_frame.data, width, height, 3*width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                painter = QPainter(pixmap)
                self.draw_rois(painter)
                painter.end()
                
                self.current_pixmap = pixmap
                self.update_display()
    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())