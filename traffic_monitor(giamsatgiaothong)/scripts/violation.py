import cv2

def check_violation(vehicle_bbox, stop_lines, light_status):
    """
    Kiểm tra xem phương tiện có vượt đèn đỏ không.

    Args:
        vehicle_bbox (list): [x1, y1, x2, y2]
        stop_lines (list): danh sách vạch dừng liên quan [[x1, y1, x2, y2], ...]
        light_status (str): 'red', 'green', 'yellow', 'unknown'

    Returns:
        bool: True nếu vượt đèn đỏ, False nếu không
    """
    if light_status != "red":
        return False  # Chỉ tính là vi phạm khi đèn đỏ

    x1, y1, x2, y2 = vehicle_bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    vehicle_center = (cx, cy)

    for line in stop_lines:
        if crossed_line(vehicle_center, line):
            return True

    return False


def crossed_line(point, line):
    """
    Kiểm tra điểm có vượt qua vạch không.

    Args:
        point (tuple): (px, py)
        line (list): [x1, y1, x2, y2]

    Returns:
        bool: True nếu vượt vạch
    """
    px, py = point
    x1, y1, x2, y2 = line

    # Giả định vạch nằm ngang (dưới dạng từ trái sang phải)
    # Nếu vehicle center nằm phía dưới vạch (py < y1), coi là vượt
    return py < min(y1, y2)


def draw_violation(frame, bbox, color=(0, 0, 255), label="Violation"):
    """
    Vẽ bounding box màu đỏ và nhãn nếu vi phạm.

    Args:
        frame (np.ndarray): Frame video
        bbox (list): [x1, y1, x2, y2]
        color (tuple): Màu RGB
        label (str): Nhãn hiển thị
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
