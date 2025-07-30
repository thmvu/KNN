from ultralytics import YOLO

model = YOLO("weights/best_1.pt")  # Thay đường dẫn nếu khác
print(model.names)
