import numpy as np

def detect_light(preds):
    """
    Phân tích kết quả dự đoán từ model đèn giao thông (YOLO)
    Trả về danh sách [(bbox, class_id, conf)], mỗi cái là 1 đèn.
    """
    results = []
    for pred in preds:
        for box in pred.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            results.append(((x1, y1, x2, y2), cls_id, conf))
    return results

def get_traffic_light_status(detections):
    """
    Trả về trạng thái đèn hiện tại dựa trên các detection.
    Ưu tiên RED > YELLOW > GREEN nếu có nhiều đèn.

    class_id:
        0 = red
        1 = yellow
        2 = green
    """
    priority = {0: 'red', 1: 'yellow', 2: 'green'}
    found_classes = set()

    for _, cls_id, conf in detections:
        if conf > 0.5:  # chỉ lấy nếu độ tự tin cao
            found_classes.add(cls_id)

    for cls in [0, 1, 2]:  # ưu tiên đỏ > vàng > xanh
        if cls in found_classes:
            return priority[cls]

    return 'unknown'
