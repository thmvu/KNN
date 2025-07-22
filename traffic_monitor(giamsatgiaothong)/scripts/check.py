from ultralytics import YOLO

# Load model .pt
model = YOLO("weights/light.pt")

# In ra số class
print(f"Số class: {model.model.nc}")

# In ra danh sách tên class
print("Tên class:")
print(model.model.names)
