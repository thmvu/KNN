from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model.train(
    data="output/Red-light-violation-detect-1/data.yaml",  # ✅ đúng path thật
    epochs=50,
    imgsz=640
)