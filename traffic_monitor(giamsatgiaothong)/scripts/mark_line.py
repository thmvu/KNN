import cv2
import json
import os

stop_lines = []  # Danh sách vạch dừng [{"line": [(x1, y1), (x2, y2)], "light_ids": ["0", "1"]}]
drawing = False
current_line = []
current_light_ids = []

def click_event(event, x, y, flags, param):
    global drawing, current_line, stop_lines, current_light_ids

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            current_line = [(x, y)]
            drawing = True
        else:
            current_line.append((x, y))
            drawing = False
            stop_lines.append({
                "line": current_line.copy(),
                "light_ids": current_light_ids.copy() if current_light_ids else []
            })
            print(f"[+] Đã lưu vạch với đèn: {current_light_ids} - {current_line}")
            current_line = []

def draw_stop_lines(frame, stop_line_file):
    global current_light_ids

    window_name = 'Draw Stop Line'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    try:
        print("=== Vẽ vạch dừng ===")
        print("Phím i: Nhập ID đèn giao thông")
        print("Phím s: Lưu stop_line.json")
        print("Phím q: Thoát")

        while True:
            temp_frame = frame.copy()
            for stop in stop_lines:
                cv2.line(temp_frame, tuple(stop["line"][0]), tuple(stop["line"][1]), (0, 0, 255), 2)
                label = ','.join(stop.get("light_ids", []))
                cv2.putText(temp_frame, label, tuple(stop["line"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if len(current_line) == 1:
                cv2.circle(temp_frame, current_line[0], 5, (255, 0, 0), -1)

            cv2.imshow(window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('i'):
                ids = input("Nhập ID các đèn (cách nhau bằng dấu phẩy): ").strip()
                current_light_ids = [id.strip() for id in ids.split(",") if id.strip()]
                print(f"[~] Sử dụng light_ids: {current_light_ids}")
            elif key == ord('s'):
                with open(stop_line_file, 'w') as f:
                    json.dump(stop_lines, f, indent=2)
                print(f"[💾] Đã lưu {len(stop_lines)} vạch vào {stop_line_file}")
            elif key == ord('q'):
                break

        cv2.destroyWindow(window_name)
    except Exception as e:
        print(f"[Lỗi] {e}")
        cv2.destroyWindow(window_name)

def load_stop_lines(file):
    if not os.path.exists(file):
        print(f"[!] Không tìm thấy file: {file}")
        return []

    try:
        with open(file, 'r') as f:
            lines = json.load(f)
            # Đảm bảo đúng format: mỗi phần tử là dict có "line" (list 2 điểm) và "light_ids" (list)
            valid_lines = []
            for item in lines:
                if "line" in item and isinstance(item["line"], list) and len(item["line"]) == 2:
                    item["light_ids"] = item.get("light_ids", [])
                    valid_lines.append(item)
            print(f"[✅] Đã load {len(valid_lines)} vạch dừng từ {file}")
            return valid_lines
    except Exception as e:
        print(f"[Lỗi khi load stop_line.json] {e}")
        return []
