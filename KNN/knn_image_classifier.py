import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = r"C:\vs code\hoctap\hoc\KNN\dataset"

def load_images_from_folder(folder):
    images, labels = [], []
    class_names = sorted(os.listdir(folder))
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path): continue
        for filename in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels, class_names

def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        _, descriptors = sift.detectAndCompute(img, None)
        descriptors_list.append(descriptors if descriptors is not None else np.array([]))
    return descriptors_list

def build_bovw_histogram(descriptors, kmeans, k):
    hist = np.zeros(k)
    if descriptors is not None and len(descriptors) > 0:
        words = kmeans.predict(descriptors)
        for w in words:
            hist[w] += 1
    return hist

if __name__ == "__main__":
    # 1. Load ảnh và nhãn
    images, labels, class_names = load_images_from_folder(DATASET_PATH)
    print(f"Tải {len(images)} ảnh từ {len(class_names)} lớp:", class_names)

    # 2. Trích xuất SIFT
    descriptors_list = extract_sift_features(images)

    # 3. Gom descriptor để tạo codebook
    all_descriptors = np.vstack([d for d in descriptors_list if d is not None and len(d) > 0])
    k = 100
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=k*20, random_state=42)
    kmeans.fit(all_descriptors)
    print(f"Đã tạo codebook với {k} visual words")

    # 4. Tạo histogram BoVW
    X_features = np.array([build_bovw_histogram(d, kmeans, k) for d in tqdm(descriptors_list, desc="Tạo BoVW vector")])
    y_labels = np.array(labels)

    # 5. Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.3, random_state=42)

    # 6. Huấn luyện K-NN
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    knn.fit(X_train, y_train)

    # 7. Dự đoán & đánh giá
    y_pred = knn.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 8. Vẽ confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
