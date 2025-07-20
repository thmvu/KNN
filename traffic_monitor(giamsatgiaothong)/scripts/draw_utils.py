import cv2
import json

def load_stop_lines(json_path):
    """
    Đọc stop_line.json → dict {light_id: [ [x1, y1, x2, y2], ... ]}
    """
    with open(json_path, "r") as f:
        return json.load(f)

def draw_stop_lines(frame, stop_lines, light_status, color_dict=None):
    """
    Vẽ các stop line lên frame theo trạng thái đèn
    """
    if color_dict is None:
        color_dict = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "unknown": (128, 128, 128)
        }
    color = color_dict.get(light_status, (128, 128, 128))

    for line in stop_lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
