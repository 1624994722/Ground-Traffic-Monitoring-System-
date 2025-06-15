import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QWidget, QVBoxLayout, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect
from PyQt5.QtGui import QColor, QFont

class SciFiButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.init_style()
        self._setup_animation()
        
    def init_style(self):
        self.setFont(QFont("Orbitron", 12, QFont.DemiBold))
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #0f2027, stop:1 #203a43);
                color: #00fff7;
                border-radius: 15px;
                padding: 15px;
                border: 2px solid #00fff7;
                min-width: 200px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #203a43, stop:1 #2c5364);
                color: #00ffff;
                border: 2px solid #00ffff;
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 255, 230, 80))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)

    def _setup_animation(self):
        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(200)
        
    def animate_click(self):
        orig = self.geometry()
        self.anim.setStartValue(orig)
        self.anim.setEndValue(QRect(orig.x()-5, orig.y()-5, orig.width()+10, orig.height()+10))
        self.anim.start()

class ParkingLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("无人机视角的交通违规行为识别")
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: #0a0a0a;")
        self.processes = {}
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(40)

        self.btn_1 = SciFiButton("启动违停识别系统")
        self.btn_2 = SciFiButton("启动超速识别系统")
        
        self.btn_1.clicked.connect(lambda: self.toggle_process("1", "违停识别.py"))
        self.btn_2.clicked.connect(lambda: self.toggle_process("2", "超速识别.py"))
        
        layout.addWidget(self.btn_1)
        layout.addWidget(self.btn_2)
        
        # 添加动态光效背景
        self.background_effect()

    def background_effect(self):
        # 可以添加更多视觉特效，如流动的光线等
        pass

    def toggle_process(self, key, script_name):
        btn = self.btn_1 if key == "1" else self.btn_2
        
        if key in self.processes:
            # 停止进程
            self.processes[key].terminate()
            self.processes[key].wait()
            del self.processes[key]
            btn.setText(f"启动{script_name[:-3]}")
            btn.setStyleSheet(btn.styleSheet().replace("#ff4757", "#00fff7").replace("停止", ""))
        else:
            # 启动进程
            try:
                self.processes[key] = subprocess.Popen([sys.executable, script_name])
                btn.setText(f"停止{script_name[:-3]}")
                btn.setStyleSheet(btn.styleSheet().replace("#00fff7", "#ff4757"))
                btn.animate_click()
            except Exception as e:
                print(f"启动失败: {e}")

    def closeEvent(self, event):
        # 确保所有子进程在主窗口关闭时终止
        for proc in self.processes.values():
            proc.terminate()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ParkingLauncher()
    window.show()
    sys.exit(app.exec_())
