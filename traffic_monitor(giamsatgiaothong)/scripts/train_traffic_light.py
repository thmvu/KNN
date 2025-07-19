from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="/kaggle/working/Traffic-Light-Detection-2/data.yaml",  
    epochs=100,
    imgsz=640,
    batch=64,
    # device='0,1,2,3',
    cache=True,
    workers=8,
    amp=True,
    freeze      = 11,              # đóng băng backbone
    lr0         = 1e-3,            # AdamW mặc định
    lrf         = 0.01, 
    optimizer   = "AdamW",
    weight_decay= 5e-5,
    patience    = 20, 
    hsv_h=0.03, hsv_s=0.6, hsv_v=0.5,
      translate=0.1, scale=0.1, shear=0.1,
      fliplr=0.5,
      mosaic=1.0, mixup=0.5, cutmix=0.5
)