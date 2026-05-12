from ultralytics import YOLO

# Load a model
model = YOLO("/root/ultralytics/runs/detect/train3/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="/root/ultralytics/ytest.yaml", imgsz=640, batch=16) 



# yolo predict model=/root/ultralytics/runs/detect/train2/weights/best.pt source='/root/ultralytics/datasets/ytest/images/test'
