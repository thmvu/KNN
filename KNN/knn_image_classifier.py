import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Cấu hình ===
DATASET_PATH = r"C:\vs code\hoctap\hoc\KNN\dataset"
  # Thư mục chứa ảnh, bên trong có 2 folder: cat và dog
IMAGE_SIZE = (64, 64)     # Kích thước resize ảnh
K = 3                     # Số lượng hàng xóm gần nhất

# === Hàm đổi tên ảnh trong folder thành 001.jpg, 002.jpg, ... ===
def rename_images(folder_path):
    files = sorted(os.listdir(folder_path))
    count = 1
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            new_name = f"{count:03d}.jpg"
            new_path = os.path.join(folder_path, new_name)
            if file_path != new_path:
                os.rename(file_path, new_path)
            count += 1

# === Đọc ảnh từ 2 folder cat và dog, resize và gán nhãn ===
def load_images_from_folder(folder_path):
    images = []
    labels = []
    label_names = ['cat', 'dog']  # Cố định 2 lớp

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {folder_path}")

    for label in label_names:
        label_folder = os.path.join(folder_path, label)
        if not os.path.isdir(label_folder):
            print(f"Bỏ qua {label_folder}: không phải thư mục")
            continue

        rename_images(label_folder)  # Đổi tên ảnh trước khi đọc

        files = sorted(os.listdir(label_folder))
        for filename in files:
            file_path = os.path.join(label_folder, filename)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                img_flatten = img.flatten()
                images.append(img_flatten)
                labels.append(label)
            else:
                print(f"Lỗi: Không thể đọc file {file_path}")

    if not images:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ trong {folder_path}")

    return np.array(images), np.array(labels)

# === Main ===
if __name__ == "__main__":
    try:
        X, y = load_images_from_folder(DATASET_PATH)
    except Exception as e:
        print(f"Lỗi khi load dữ liệu: {e}")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
