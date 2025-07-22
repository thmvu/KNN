import cv2
from detect_traffic_light import load_model as load_light_model, detect as detect_light

# ===== Đường dẫn video / ảnh =====
VIDEO_PATH = "input/videos/videoplayback3.mp4"  # hoặc đổi sang ảnh

# ===== Load model =====
light_model = load_light_model()

# ===== Mở video và lấy frame đầu tiên =====
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Không đọc được frame từ video.")
    exit()

# ===== Detect traffic lights =====
light_detections = detect_light(light_model, frame)

# ===== Gán ID từ trái sang phải nếu chưa có =====
sorted_lights = sorted(light_detections, key=lambda d: (d["box"][0] + d["box"][2]) // 2)
for idx, det in enumerate(sorted_lights):
    det["id"] = f"light_{idx}"

# ===== In ra thông tin =====
print("📋 Danh sách đèn giao thông phát hiện được:")
for light in sorted_lights:
    x1, y1, x2, y2 = light["box"]
    print(f"- ID: {light['id']}, Status: {light['status']}, BBox: ({x1}, {y1}, {x2}, {y2})")

# ===== Vẽ lên frame để xem trực tiếp =====
for light in sorted_lights:
    x1, y1, x2, y2 = light["box"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{light['id']} ({light['status']})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Traffic Light ID Check", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
