import os
import cv2
import json
from datetime import datetime

from detect_vehicle import load_model as load_vehicle_model, detect as detect_vehicle
from detect_traffic_light import load_model as load_light_model, detect as detect_light, draw_lights
from mark_line import load_stop_lines, draw_stop_lines

# Đường dẫn
VIDEO_PATH = "input/videos/videoplayback.mp4"
STOPLINE_DIR = "stopline"
STOPLINE_PATH = os.path.join(STOPLINE_DIR, "stop_line.json")
OUTPUT_VIDEO_PATH = "output/result.mp4"
VIOLATION_DIR = "violations"

# Tạo thư mục cần thiết
os.makedirs(STOPLINE_DIR, exist_ok=True)
os.makedirs(VIOLATION_DIR, exist_ok=True)

# Load models
print("🔍 Load YOLO models...")
vehicle_model = load_vehicle_model()
light_model = load_light_model()

# Mở video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Load hoặc vẽ stopline
if not os.path.exists(STOPLINE_PATH):
    print("🖍️ Chưa có stop_line.json, khởi động vẽ...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame đầu.")
        exit(1)
    draw_stop_lines(frame, STOPLINE_PATH)

stop_lines = load_stop_lines(STOPLINE_PATH)

frame_index = 0
print("▶️ Bắt đầu xử lý video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect traffic lights & vehicles
    light_detections = detect_light(light_model, frame)
    vehicle_detections = detect_vehicle(vehicle_model, frame)

    # Ánh xạ trạng thái đèn
    light_status_map = {
        light["id"]: {"status": light["status"], "box": light["box"]}
        for light in light_detections
    }

    # Vẽ đèn
    frame = draw_lights(frame, light_detections)

    # Vẽ stopline
    for line in stop_lines:
        color = (255, 255, 255)  # default: trắng
        for lid in line["light_ids"]:
            light_info = light_status_map.get(lid)
            if light_info and light_info["status"] == "red":
                color = (0, 0, 255)  # đỏ khi đèn đỏ
        pts = line["points"]
        for i in range(0, len(pts), 2):
            p1, p2 = tuple(pts[i]), tuple(pts[i+1])
            cv2.line(frame, p1, p2, color, 2)

    # Check violation
    for veh in vehicle_detections:
        x1, y1, x2, y2 = veh["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for line in stop_lines:
            for lid in line["light_ids"]:
                light_info = light_status_map.get(lid)
                if not light_info or light_info["status"] != "red":
                    continue

                for i in range(0, len(line["points"]), 2):
                    x_a, y_a = line["points"][i]
                    x_b, y_b = line["points"][i + 1]

                    # Kiểm tra điểm center phương tiện đã vượt chưa
                    if x_b - x_a == 0:
                        # Đường thẳng đứng
                        if cx > x_a:
                            cv2.putText(frame, "VIOLATION", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                            cv2.imwrite(os.path.join(VIOLATION_DIR, filename), frame)
                    else:
                        # Tính khoảng cách từ điểm đến đoạn thẳng
                        slope = (y_b - y_a) / (x_b - x_a)
                        dist = abs(slope * cx - cy + y_a - slope * x_a) / (slope**2 + 1)**0.5
                        if dist > 10 and cy > min(y_a, y_b):  # vượt qua line
                            cv2.putText(frame, "VIOLATION", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                            cv2.imwrite(os.path.join(VIOLATION_DIR, filename), frame)

    out.write(frame)
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Xử lý hoàn tất.")
