import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Cấu hình ===
DATASET_PATH = "."  # Thư mục chứa ảnh (KNN folder with cat and dog subfolders)
IMAGE_SIZE = (64, 64)  # Resize ảnh về kích thước đồng đều
K = 3  # Số lượng hàng xóm gần nhất

# === Đọc ảnh và gán nhãn ===
def load_images_from_folder(folder_path):
    images = []
    labels = []

    # Kiểm tra thư mục tồn tại
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    label_names = os.listdir(folder_path)
    for label in label_names:
        label_folder = os.path.join(folder_path, label)
        if not os.path.isdir(label_folder):
            continue

        for filename in os.listdir(label_folder):
            file_path = os.path.join(label_folder, filename)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                img_flatten = img.flatten()  # Biến ảnh thành vector 1 chiều
                images.append(img_flatten)
                labels.append(label)

    if not images:
        raise ValueError(f"No valid images found in {folder_path}")

    return np.array(images), np.array(labels)

# === Load dữ liệu ===
try:
    X, y = load_images_from_folder(DATASET_PATH)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# === Chia dữ liệu train/test ===
if len(X) == 0 or len(y) == 0:
    print("Error: No data to process. Check your dataset folder.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Huấn luyện mô hình KNN ===
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)

# === Dự đoán và đánh giá ===
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))