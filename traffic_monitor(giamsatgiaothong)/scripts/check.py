from ultralytics import YOLO

model = YOLO("weights/light1.pt")
names = model.names  # dict: {class_id: class_name}
print(f"Số class: {len(names)}")
print("Tên class:")
print(names)
