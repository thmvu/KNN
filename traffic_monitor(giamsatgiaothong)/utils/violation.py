def check_violation(light_label, vehicle_bbox, stop_line_y):
    _, _, _, y2 = vehicle_bbox
    if light_label == "red_light" and y2 > stop_line_y:
        return True
    return False
