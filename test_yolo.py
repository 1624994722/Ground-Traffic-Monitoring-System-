import cv2
from ultralytics import YOLO
import os
import yaml

# 配置路径
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKSPACE, 'runs', 'detect', 'train', 'weights', 'best.pt')
TEST_VIDEO_PATH = os.path.join(WORKSPACE, 'videos', 'traffic.flv')  # 可替换为VisDrone测试图片路径

# 加载模型
model = YOLO(MODEL_PATH)

# 从 VisDrone.yaml 读取类别名称
with open(os.path.join(WORKSPACE, 'VisDrone.yaml'), 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
class_names = data['names']

# 推理函数
def test_yolo():
    # 初始化视频捕获
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    frame_count = 0
    
    # 创建保存目录
    SAVE_DIR = os.path.join(WORKSPACE, 'detect', 'predict')
    os.makedirs(SAVE_DIR, exist_ok=True)

    while cap.isOpened():
        # 读取帧
        success, frame = cap.read()
        if not success:
            break

        # 模型推理
        results = model(frame)

        # 可视化结果
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
            clss = result.boxes.cls.cpu().numpy()    # 类别索引
            confs = result.boxes.conf.cpu().numpy()  # 置信度

            for box, cls, conf in zip(boxes, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls)
                label = f'{class_names.get(cls_id, "unknown")} {conf:.2f}'

                # 绘制边界框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存结果
        save_path = os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(save_path, frame)
        frame_count += 1

        # 显示结果
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_yolo()