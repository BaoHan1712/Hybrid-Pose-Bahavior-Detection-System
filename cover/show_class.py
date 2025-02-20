from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("model/yolo11n-pose.pt")

# print(model.names) 

model.predict(r"data\v2.mp4", show=True, save=True)
# model.export(format="engine",imgsz=640,dynamic=True, half=True)