from ultralytics import YOLO

def load_model():
    return YOLO("weights/best_1.pt")  # Đường dẫn model đã train

def detect(model, frame, conf_thresh=0.5, iou_thresh=0.5, target_classes=None):
    results = model.track(source=frame, persist=True, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    
    if results.boxes is None:
        return []

    bboxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        label = model.names[cls_id]

        # Nếu có target_classes thì lọc theo nhãn
        if target_classes and label not in target_classes:
            continue

        bboxes.append({
            "id": track_id,
            "box": (x1, y1, x2, y2),
            "label": label,
            "conf": conf
        })

    return bboxes
