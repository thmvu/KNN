# 🚦 Hệ Thống Giám Sát Giao Thông - Phát Hiện Vượt Đèn Đỏ

Dự án này xây dựng một hệ thống giám sát giao thông thông minh, sử dụng mô hình **YOLOv11** để **phát hiện phương tiện vượt đèn đỏ** trong video. Hệ thống hoạt động bằng cách nhận diện đèn giao thông, phương tiện, vạch dừng và tự động lưu lại bằng chứng khi vi phạm xảy ra.

---

## 🧠 Công Nghệ Sử Dụng

- **Ngôn ngữ lập trình:** Python 3.x  
- **Mô hình AI:** [YOLOv11](https://github.com/ultralytics/ultralytics) (phát hiện vật thể theo thời gian thực)
- **Thư viện chính:**
  - `ultralytics` (YOLOv11)
  - `OpenCV` (xử lý ảnh/video)
  - `NumPy` (xử lý ma trận)
  - `CSV`, `json`, `datetime` (ghi log và xử lý dữ liệu)
- **Giao diện người dùng vẽ vạch dừng:**
- OpenCV + chuột

---

## ✅ Chức Năng Chính

- 🚗 Nhận diện và theo dõi **phương tiện giao thông** bằng YOLOv8.
- 🚦 Phân loại và nhận diện **trạng thái đèn giao thông** (Đỏ / Vàng / Xanh).
- ✏️ Hỗ trợ **vẽ vạch dừng** thủ công cho từng ID đèn giao thông.
- ❌ Phát hiện phương tiện **vượt vạch khi đèn đỏ**.
- 🖼️ **Lưu ảnh bằng chứng** và ghi log vi phạm vào file CSV.
- 🎥 Xuất video có overlay đầy đủ: bounding box, ID, trạng thái đèn, vạch dừng.

---

## 📁 Cấu Trúc Thư Mục
```
traffic_monitoring/
├── input/
│ └── videos/ # Video đầu vào
├── output/
│ ├── videos/ # Video đầu ra đã xử lý
│ ├── violations/ # Ảnh phương tiện vi phạm
│ └── violation.csv # Log chi tiết vi phạm
├── models/
│ ├── vehicle.pt # Mô hình YOLOv8 phát hiện xe
│ └── traffic_light.pt # Mô hình YOLOv8 phát hiện đèn giao thông
├── stop_line/
│ └── stop_line.json # Vạch dừng gắn với đèn giao thông
├── mark_line.py # Công cụ vẽ vạch dừng
├── main.py # Chạy hệ thống chính
├── detect_vehicle.py # Module nhận diện phương tiện
├── detect_traffic_light.py # Module nhận diện đèn giao thông
├── violation.py # Kiểm tra & ghi nhận vi phạm
└── requirements.txt # Danh sách thư viện cần cài
```


---

## ⚙️ Cài Đặt

1. **Clone dự án:**

```

git clone https://github.com/yourname/traffic-monitoring.git
cd traffic-monitoring
Tạo virtual environment (khuyên dùng):
```
```

python -m venv venv
venv\Scripts\activate  # Trên Windows
# hoặc
source venv/bin/activate  # Trên Linux/Mac
Cài đặt thư viện cần thiết:

pip install -r requirements.txt
Tải mô hình YOLOv8 và đặt vào thư mục models/:

vehicle.pt: mô hình nhận diện xe.

traffic_light.pt: mô hình nhận diện trạng thái đèn giao thông.
```
🧪 Hướng Dẫn Sử Dụng
**1. Vẽ Vạch Dừng**
Chạy lệnh:

```
python mark_line.py
```
Video sẽ hiện khung hình đầu tiên.

Dùng chuột vẽ vạch dừng.

Nhập ID đèn giao thông tương ứng (VD: light_0).

Nhấn s để lưu vạch vào stop_line/stop_line.json.

2. Chạy Hệ Thống Chính
```
python main.py
```
Hệ thống sẽ tự động:

Nhận diện đèn giao thông & xe.

Kiểm tra trạng thái đèn.

So sánh vị trí xe với vạch dừng.

Ghi lại các phương tiện vượt đèn đỏ.

📤 Kết Quả Xuất Ra
output/videos/*.mp4: video có box, vạch dừng, trạng thái đèn.

output/violations/*.jpg: ảnh các xe vi phạm.

output/violation.csv: log chi tiết, ví dụ:

```
vehicle_id,frame_number,filename
3,157,violations/vehicle_3_frame_157.jpg
7,240,violations/vehicle_7_frame_240.jpg
```
🔭 Định Hướng Mở Rộng
Nhận diện biển số xe (ALPR).

Xử lý camera trực tiếp (live stream).

Triển khai dashboard web để giám sát từ xa.

Tích hợp với cơ sở dữ liệu thành phố.

📮 Liên Hệ
Tác giả: Vũ Từ
Nếu bạn muốn đóng góp, cải tiến hoặc cần hỗ trợ, hãy liên hệ qua GitHub hoặc email cá nhân.

📝 Giấy Phép
Dự án sử dụng cho mục đích học tập và nghiên cứu. Không chịu trách nhiệm nếu sử dụng sai mục đích.
