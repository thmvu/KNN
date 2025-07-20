import cv2

def detect_vehicle(preds, class_names=None, conf_threshold=0.3):
    """
    Phân tích kết quả dự đoán từ mô hình YOLO để phát hiện phương tiện.

    Args:
        preds: Kết quả dự đoán từ model.predict().
        class_names (list): Danh sách tên lớp (ví dụ ['car', 'motorbike']).
        conf_threshold (float): Ngưỡng confidence để lọc kết quả yếu.

    Returns:
        List[dict]: Danh sách các object phát hiện dạng:
            {
                'bbox': (x1, y1, x2, y2),
                'class_id': int,
                'class_name': str,
                'confidence': float
            }
    """
    results = []

    for pred in preds:
        boxes = pred.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            class_id = int(box.cls[0])
            class_name = class_names[class_id] if class_names else str(class_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int pixel coordinates

            results.append({
                'bbox': (x1, y1, x2, y2),
                'class_id': class_id,
                'class_name': class_name,
                'confidence': conf
            })

    return results


def draw_vehicle_detections(frame, detections, color=(0, 255, 0)):
    """
    Vẽ bounding boxes và nhãn lên khung hình.

    Args:
        frame: Ảnh đầu vào (numpy array - BGR).
        detections: List các detection từ detect_vehicle().
        color: Màu RGB để vẽ bbox.

    Returns:
        frame: Ảnh đầu ra sau khi đã vẽ.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Vẽ khung
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Vẽ nhãn
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame
