from roboflow import Roboflow

# 👉 Thay bằng API KEY của bạn
ROBOFLOW_API_KEY = "6jPBN33eSFxcNmHr9NqD"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# Dataset 1: Detect phương tiện vi phạm
project_vehicle = rf.workspace("ti-aqt3w").project("red-light-violation-detect-795dz")
project_vehicle.version(1).download("yolov8")

# Dataset 2: Detect đèn đỏ/xanh
project_light = rf.workspace("ti-aqt3w").project("traffic-light-gxodz-qw7mv")
project_light.version(1).download("yolov8")

print("✅ Datasets đã được tải về thành công!")
