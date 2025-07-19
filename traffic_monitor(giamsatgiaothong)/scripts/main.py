import os
import cv2
import json
from datetime import datetime
from detect_vehicle import load_model as load_vehicle_model, detect as detect_vehicle
from detect_traffic_light import load_model as load_light_model, detect as detect_light
from mark_line import draw_stop_lines, load_stop_lines

def get_stopline_path(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    return f"stop_lines/{name}.json"

def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def cross_product(p1, p2, p):
    x1, y1 = p1
    x2, y2 = p2
    x, y = p
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def ensure_dirs():
    os.makedirs("stop_lines", exist_ok=True)
    os.makedirs("output/violations", exist_ok=True)

def inspect_lights_on_first_frame(video_path, light_model):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print("Không thể đọc frame đầu tiên.")
        return []

    results = light_model(frame)
    boxes = results[0].boxes
    print(f"📸 [DETECT_LIGHT] Số lượng object phát hiện: {len(boxes)}\n")
    print("💡 [Thông tin đèn giao thông ở frame đầu]:")

    light_infos = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = light_model.names[cls_id]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        print(f" - ID: {idx}, Nhãn: {cls_name}, Box: ({x1}, {y1}), ({x2}, {y2}), Tâm: ({cx}, {cy})")

        light_infos.append({
            "id": idx,
            "box": [x1, y1, x2, y2],
            "status": cls_name
        })

    return light_infos


def main(video_path):
    ensure_dirs()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_path}")
        return

    stopline_path = get_stopline_path(video_path)

    # Load model đèn
    light_model = load_light_model()
    first_lights = inspect_lights_on_first_frame(video_path, light_model)

    # Nếu chưa có vạch dừng thì vẽ
    if not os.path.exists(stopline_path):
        print("📌 Vẽ vạch dừng lần đầu...")
        ret, frame0 = cap.read()
        if not ret:
            print("❌ Không thể đọc frame đầu.")
            return
        draw_stop_lines(frame0, stopline_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Load thông tin vạch dừng
    stop_lines = load_stop_lines(stopline_path)
    print(f"✅ Đã load {len(stop_lines)} vạch dừng từ {stopline_path}")

    # Load model xe
    vehicle_model = load_vehicle_model()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Nhận diện xe và đèn
        vehicles = detect_vehicle(vehicle_model, frame)
        lights = detect_light(light_model, frame)

        # Tạo dict: id -> trạng thái đèn
        light_status = {light["id"]: light["status"] for light in lights}

        for idx, stopline in enumerate(stop_lines):
            p1, p2 = tuple(stopline["line"][0]), tuple(stopline["line"][1])
            linked_ids = stopline.get("light_ids", [])

            is_red = any(light_status.get(lid) == "red" for lid in linked_ids)

            line_color = (0, 0, 255) if is_red else (0, 255, 0)
            cv2.line(frame, p1, p2, line_color, 2)

            if is_red:
                for box in vehicles:
                    cx, cy = center_of_box(box)
                    if cross_product(p1, p2, (cx, cy)) < 0:
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        filename = f"output/violations/violation_{frame_idx}_{idx}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"[⚠️ VI PHẠM] Frame {frame_idx}, vạch {idx}, ảnh: {filename}")

        cv2.putText(frame, f"Frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Giám sát đèn đỏ", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "input/videos/videoplayback.mp4"
    main(video_path)
