import cv2
import os
import csv
from datetime import datetime

violation_memory = {}  # {vehicle_id: {'bbox':..., 'saved': bool, 'frame': int, 'cooldown': int}}


def point_line_distance(px, py, x1, y1, x2, y2):
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = px - xx
    dy = py - yy
    return (dx * dx + dy * dy) ** 0.5, dy


def check_violation(vehicle_id, vehicle_bbox, stop_lines, light_status,
                    frame, frame_number, save_dir="output/violations", threshold=25, cooldown_frames=15):
    if light_status != "red" or frame is None:
        return False

    x1, y1, x2, y2 = vehicle_bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    for line in stop_lines:
        for i in range(0, len(line), 4):
            lx1, ly1, lx2, ly2 = line[i:i+4]
            dist, dy = point_line_distance(cx, cy, lx1, ly1, lx2, ly2)

            if dist < threshold and dy > 0:
                # Ghi nhớ vi phạm
                if vehicle_id not in violation_memory:
                    violation_memory[vehicle_id] = {
                        'bbox': vehicle_bbox,
                        'saved': False,
                        'frame': frame_number,
                        'cooldown': cooldown_frames
                    }
                else:
                    violation_memory[vehicle_id].update({
                        'bbox': vehicle_bbox,
                        'frame': frame_number,
                        'cooldown': cooldown_frames
                    })

                # Lưu ảnh nếu chưa lưu
                if not violation_memory[vehicle_id]['saved']:
                    try:
                        os.makedirs(save_dir, exist_ok=True)

                        full_path = os.path.join(save_dir, f"violation_{vehicle_id}_frame{frame_number}.jpg")
                        success = cv2.imwrite(full_path, frame)
                        if not success:
                            print(f"[ERROR] Không thể lưu ảnh tại: {full_path}")
                            return False
                        else:
                            print(f"[INFO] Đã lưu ảnh vi phạm: {full_path}")

                        # Lưu log CSV
                        csv_path = os.path.join(save_dir, "violation.csv")
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if f.tell() == 0:
                                writer.writerow(["vehicle_id", "frame_number", "datetime"])
                            writer.writerow([vehicle_id, frame_number, datetime.now().isoformat()])

                        violation_memory[vehicle_id]['saved'] = True

                    except Exception as e:
                        print(f"[EXCEPTION] Lỗi khi lưu ảnh/CSV: {e}")
                        return False

                return True

    return False


def draw_violation(frame, bbox, label="VIOLATION"):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def update_violation_memory():
    expired_ids = []
    for vehicle_id, data in violation_memory.items():
        if data['cooldown'] > 0:
            data['cooldown'] -= 1
        else:
            expired_ids.append(vehicle_id)
    for vid in expired_ids:
        violation_memory.pop(vid)
