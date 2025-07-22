import os
import cv2
import json
import numpy as np
from datetime import datetime
import csv

from detect_vehicle import load_model as load_vehicle_model, detect as detect_vehicle
from detect_traffic_light import load_model as load_light_model, detect as detect_light, draw_lights
from mark_line import load_stop_lines, draw_stop_lines
from violation import check_violation, draw_violation, update_violation_memory, violation_memory

# ==== Đường dẫn video ====
VIDEO_PATH = "input/videos/videoplayback.mp4"
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

STOPLINE_DIR = "stopline"
STOPLINE_PATH = os.path.join(STOPLINE_DIR, f"{VIDEO_NAME}_stopline.json")
OUTPUT_VIDEO_PATH = f"output/{VIDEO_NAME}_result.mp4"
VIOLATION_DIR = "output/violations"
VIOLATION_LOG = os.path.join(VIOLATION_DIR, "violation.csv")

# ==== Tạo thư mục nếu chưa có ====
os.makedirs(STOPLINE_DIR, exist_ok=True)
os.makedirs(VIOLATION_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

# ==== Load Models ====
print("🔍 Load YOLO models...")
vehicle_model = load_vehicle_model()
light_model = load_light_model()

# ==== Đọc video ====
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WAIT_FRAMES = 30

print("🔧 In ra ID các đèn giao thông để bạn gán stop line...")

for i in range(WAIT_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    light_detections = detect_light(light_model, frame)
    for det in light_detections:
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) // 2
        if cx < width // 3:
            det["id"] = "left"
        elif cx > 2 * width // 3:
            det["id"] = "right"
        else:
            det["id"] = "center"

    print(f"\n[Frame {i}]")
    for det in light_detections:
        print(f"  -> ID: {det['id']}, Box: {det['box']}, Status: {det['status']}")

cap.release()

# ==== Vẽ Stop Line nếu chưa có ====
if not os.path.exists(STOPLINE_PATH):
    print(f"🖍️ Không tìm thấy {STOPLINE_PATH}, cần vẽ lại stop line...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame đầu tiên để vẽ stop line.")
        exit(1)
    draw_stop_lines(frame, STOPLINE_PATH)
    cap.release()

# ==== Load Stop Line ====
stop_lines = load_stop_lines(STOPLINE_PATH)

# ==== Bắt đầu xử lý video ====
print("▶️ Bắt đầu xử lý video...")
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

frame_index = 0
violated_ids = set()

# ==== Ghi log CSV ====
with open(VIOLATION_LOG, 'w', newline='') as log_file:
    writer = csv.writer(log_file)
    writer.writerow(["vehicle_id", "frame_number", "filename"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Detect traffic light ---
        light_detections = detect_light(light_model, frame)
        for det in light_detections:
            x1, y1, x2, y2 = det["box"]
            cx = (x1 + x2) // 2
            if cx < width // 3:
                det["id"] = "left"
            elif cx > 2 * width // 3:
                det["id"] = "right"
            else:
                det["id"] = "center"

        # --- Detect vehicle (with tracking ID) ---
        vehicle_detections = detect_vehicle(vehicle_model, frame)

        # --- Light ID to status map ---
        light_status_map = {
            light["id"]: {"status": light["status"], "box": light["box"]}
            for light in light_detections
        }

        # --- Vẽ đèn ---
        frame = draw_lights(frame, light_detections)

        # --- Vẽ stop line ---
        for line in stop_lines:
            color = (255, 255, 255)
            for lid in line["light_ids"]:
                light_info = light_status_map.get(lid)
                if light_info and light_info["status"] == "red":
                    color = (0, 0, 255)
            pts = line["points"]
            for i in range(0, len(pts), 2):
                p1, p2 = tuple(pts[i]), tuple(pts[i + 1])
                cv2.line(frame, p1, p2, color, 2)

        # --- Kiểm tra vi phạm ---
            for veh in vehicle_detections:
                x1, y1, x2, y2 = veh["box"]
                bbox = [x1, y1, x2, y2]
                track_id = veh.get("id", -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if track_id in violated_ids:
                    draw_violation(frame, bbox)
                    continue

                for line in stop_lines:
                    for lid in line["light_ids"]:
                        light_info = light_status_map.get(lid)
                        if not light_info:
                            continue
                        status = light_info["status"]

                        for i in range(0, len(line["points"]), 2):
                            p1 = line["points"][i]
                            p2 = line["points"][i + 1]
                            line_seg = [p1[0], p1[1], p2[0], p2[1]]

                            vehicle_id = f"{track_id}"
                            violation_result = check_violation(
                                vehicle_id, bbox, [line_seg], status,
                                frame.copy(), frame_index, save_dir=VIOLATION_DIR
                            )
                            if violation_result:
                                draw_violation(frame, bbox)
                                violated_ids.add(track_id)
                                writer.writerow([vehicle_id, frame_index, violation_result])
                                break

        # ✅ Sửa lỗi: Gửi danh sách ID hiện tại vào hàm cập nhật
        current_vehicle_ids = [veh["id"] for veh in vehicle_detections if veh["id"] != -1]
        update_violation_memory(current_vehicle_ids)

        out.write(frame)
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

# === Phần kết thúc giữ nguyên ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Xử lý hoàn tất.")