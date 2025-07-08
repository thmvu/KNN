from ultralytics import YOLO
import cv2
import os

# Load model đã train
model = YOLO("yolo11s.pt")

# Mở video đầu vào
cap = cv2.VideoCapture("input/videos/videoplayback.mp4")

# Đọc một frame để lấy kích thước video
ret, frame = cap.read()
if not ret:
    print("❌ Không đọc được video đầu vào.")
    exit()

# Lấy kích thước khung hình
frame_height, frame_width = frame.shape[:2]

# Tạo thư mục output nếu chưa có
os.makedirs("output", exist_ok=True)

# Khởi tạo video writer để lưu video kết quả
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho mp4
out = cv2.VideoWriter("output/result.mp4", fourcc, 20.0, (frame_width, frame_height))

# Quay lại frame đầu
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Duyệt từng frame để nhận diện và ghi ra video
frame_count = 0
def detect_traffic_light_color(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # Ngưỡng màu đỏ
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    red_mask = red1 + red2

    # Ngưỡng màu vàng
    yellow_mask = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))

    # Ngưỡng màu xanh
    green_mask = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

    # Tính số pixel của mỗi màu
    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)

    # Chọn màu có pixel nhiều nhất
    max_count = max(red_count, yellow_count, green_count)
    if max_count == red_count:
        return "RED"
    elif max_count == yellow_count:
        return "YELLOW"
    else:
        return "GREEN" # In ra tên các lớp nhận diện
while True:

    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (640, 640))  # Đảm bảo kích thước khung hình
    # Nhận diện bằng YOLO
    results = model(frame, conf=0.55,iou=0.55)[0]
    boxes = results.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

    for i, class_id in enumerate(class_ids):
        if class_id == 9:
            print("Đang nhận diện đèn giao thông...")
            x1, y1, x2, y2 = coords[i].astype(int)

            # Cắt vùng ảnh của đèn giao thông
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            # Phát hiện màu
            color = detect_traffic_light_color(cropped)
            print("màu đèn", color)
            # Ghi nhãn lên frame
            label = f"{color}"
            if color == "RED":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif color == "YELLOW": 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            elif color == "GREEN":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if class_id ==2:
            x1, y1, x2, y2 = coords[i].astype(int)
            label =" Vehicle"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if class_id == 7:
            x1, y1, x2, y2 = coords[i].astype(int)  
            label = "truck"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out.write(frame)
# Giải phóng bộ nhớ
cap.release()
out.release()

print("✅ Xong! Video kết quả đã được lưu vào: output/result.mp4")
