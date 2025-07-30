from ultralytics import YOLO

# Hàm load mô hình YOLO từ file đã huấn luyện
def load_model():
    return YOLO("weights/best_1.pt")  # ✅ Đường dẫn tới model YOLO đã huấn luyện

# Hàm detect để phát hiện và theo dõi đối tượng trong một frame
def detect(model, frame, conf_thresh=0.4, iou_thresh=0.4, target_classes=None):
    # 📦 Dự đoán và tracking trên frame, trả về kết quả đầu tiên
    results = model.track(
        source=frame,        # Frame ảnh đầu vào
        persist=True,        # Giữ ID của object giữa các frame
        conf=conf_thresh,    # Ngưỡng confidence
        iou=iou_thresh,      # Ngưỡng IOU để nén box (NMS)
        verbose=False        # Không in log từ YOLO
    )[0]

    # ❌ Không có kết quả nào (không có box)
    if results.boxes is None:
        print("[Detect] ❌ No detections.")
        return []

    bboxes = []          # Danh sách bbox sau khi xử lý
    class_count = {}     # Thống kê số lượng object theo label

    # Duyệt từng box phát hiện được
    for box in results.boxes:
        # Trích toạ độ bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Xác suất confidence
        conf = float(box.conf[0])

        # ID lớp (class)
        cls_id = int(box.cls[0])

        # ID của object được theo dõi qua frame (nếu có)
        track_id = int(box.id[0]) if box.id is not None else -1

        # Tên nhãn (label) từ mô hình YOLO
        label = model.names[cls_id]

        # 🎯 Nếu có lọc nhãn theo target_classes thì bỏ qua nếu không trùng
        if target_classes and label not in target_classes:
            continue

        # ✅ Lưu kết quả vào danh sách
        bboxes.append({
            "id": track_id,                 # ID tracking
            "box": (x1, y1, x2, y2),        # Toạ độ bbox
            "label": label,                 # Nhãn
            "conf": conf                    # Độ tự tin
        })

        # 📊 Cập nhật đếm số lượng từng nhãn
        class_count[label] = class_count.get(label, 0) + 1

    # 🔍 In log ra terminal nếu có object nào
    if class_count:
        class_log = ', '.join(f"{v} {k}" for k, v in class_count.items())
        print(f"[Detect] ✅ {class_log}")
    else:
        print("[Detect] ⚠️ No valid detections after filtering.")

    return bboxes
