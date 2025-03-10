from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("model\yolo11m-pose.engine")

# print(model.names) 

model.predict(r"image.png", show=True, save=True)
# model.export(format="engine",imgsz=640,dynamic=True, half=True)