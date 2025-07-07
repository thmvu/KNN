from ultralytics import YOLO
import cv2
from utils.drawing import draw_box
from utils.violation import check_violation

vehicle_model = YOLO("weights/vehicle_best.pt")
light_model = YOLO("weights/traffic_light_best.pt")

cap = cv2.VideoCapture("input/videos/test_red_light.mp4")
stop_line_y = 400  # điều chỉnh theo video

while True:
    ret, frame = cap.read()
    if not ret: break

    v_res = vehicle_model(frame)[0]
    l_res = light_model(frame)[0]

    for lv in l_res.boxes:
        lx1, ly1, lx2, ly2 = lv.xyxy[0]
        light_label = l_res.names[int(lv.cls)]
        break
    else:
        light_label = None

    for vb in v_res.boxes:
        vehicle_bbox = vb.xyxy[0]
        conf = vb.conf[0]
        if conf < 0.5: continue

        is_v = check_violation(light_label, vehicle_bbox, stop_line_y)
        color = (0, 0, 255) if is_v else (0, 255, 0)
        draw_box(frame, vehicle_bbox, 
                 f"{'VIOLATION' if is_v else 'OK'} {light_label}", color)

    cv2.imshow("Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
