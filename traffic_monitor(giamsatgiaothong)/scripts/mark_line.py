import cv2
import json
import os
import sys

# Biến toàn cục cho vẽ
stop_lines = []
temp_lines = []
start_point = None
drawing = False
current_light_id = ""

def draw_stop_lines(frame, output_path):
    """
    Hàm vẽ stop line trên một frame (thường là frame đầu video),
    cho phép gán với nhiều light_id và lưu vào file JSON.
    """
    global stop_lines, temp_lines, start_point, drawing, current_light_id
    stop_lines = []
    temp_lines = []
    start_point = None
    drawing = False
    current_light_id = ""

    clone = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        global drawing, start_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            temp_lines.append((start_point, end_point))
            print(f"🖍️ Vẽ: {start_point} → {end_point}")

    cv2.namedWindow("Draw Stop Lines")
    cv2.setMouseCallback("Draw Stop Lines", mouse_callback)

    print("\n🚦 *** HƯỚNG DẪN VẼ STOP LINE ***")
    print(" - Click trái để vẽ từng đoạn thẳng.")
    print(" - Nhấn 'i' để nhập light_id và gán các đoạn vừa vẽ.")
    print(" - Nhấn 'u' để undo đoạn vừa vẽ.")
    print(" - Nhấn 'r' để reset các đoạn vẽ tạm.")
    print(" - Nhấn 's' để lưu và thoát.")
    print(" - Nhấn 'ESC' để thoát KHÔNG lưu.\n")

    while True:
        temp = clone.copy()
        # Vẽ các đoạn tạm thời đang vẽ
        for pt1, pt2 in temp_lines:
            cv2.line(temp, pt1, pt2, (0, 255, 0), 2)
        # Vẽ các đoạn đã gán vào stop_lines
        for line in stop_lines:
            pts = line["points"]
            for i in range(0, len(pts), 2):
                pt1, pt2 = tuple(pts[i]), tuple(pts[i+1])
                cv2.line(temp, pt1, pt2, (255, 0, 0), 2)
        cv2.imshow("Draw Stop Lines", temp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('u'):
            if temp_lines:
                temp_lines.pop()
                print("↩️  Đã undo.")
        elif key == ord('r'):
            temp_lines.clear()
            print("🔁 Reset các đoạn vẽ tạm.")
        elif key == ord('i'):
            current_light_id = input("💡 Nhập ID đèn giao thông (VD: left hoặc left,right): ").strip()
            if temp_lines:
                stop_lines.append({
                    "light_ids": [id.strip() for id in current_light_id.split(",")],
                    "points": [list(pt) for line in temp_lines for pt in line]
                })
                print(f"✅ Gán {len(temp_lines)} đoạn cho light_id: {current_light_id}")
                temp_lines.clear()
            else:
                print("⚠️  Chưa có đoạn nào để gán.")
        elif key == ord('s'):
            with open(output_path, "w") as f:
                json.dump(stop_lines, f, indent=4)
            print(f"💾 Đã lưu stop lines vào {output_path}")
            break
        elif key == 27:
            print("❌ Thoát KHÔNG lưu.")
            break

    cv2.destroyAllWindows()


def load_stop_lines(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

# Nếu chạy trực tiếp file này: `python mark_line.py video.mp4`
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Cách dùng: python mark_line.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Không thể đọc frame đầu từ video.")
        sys.exit(1)

    output_json = os.path.join("stopline", os.path.splitext(os.path.basename(video_path))[0] + "_stopline.json")
    os.makedirs("stopline", exist_ok=True)
    draw_stop_lines(frame, output_json)
