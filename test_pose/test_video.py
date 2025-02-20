import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import timm


class EfficientNetB2_Model(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.6):
        super(EfficientNetB2_Model, self).__init__()

        # Load EfficientNet-B2 với pretrained weights
        self.efficientnet = timm.create_model("efficientnet_b2", pretrained=True)

        # Đóng băng toàn bộ mô hình trước, chỉ fine-tune phần cuối
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Mở khóa fine-tune từ 5 block cuối thay vì chỉ 3 block
        for param in self.efficientnet.blocks[-5:].parameters():
            param.requires_grad = True

        # Lấy số feature đầu ra từ EfficientNet-B2
        num_ftrs = self.efficientnet.classifier.in_features

        # Thay thế fully connected layer bằng custom head với dropout hợp lý
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Khởi tạo trọng số
        for m in self.efficientnet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.efficientnet(x)



# Khởi tạo model YOLO và efficientnet B2
model_pose = YOLO("model/yolo11m-pose.engine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_classify = EfficientNetB2_Model(num_classes=2).to(device)
model_classify.load_state_dict(torch.load("best_efficientnetb2.pth"))
model_classify.eval()

# Transform cho model phân loại
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# Các thông số khác giữ nguyên
region_points =  [(633, 141), (992, 259), (872, 717), (-9, 658)]
cap = cv2.VideoCapture(r"data/v4.mp4")

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), 
    (5, 11), (6, 12), (11, 12), 
    (11, 13), (13, 15), (12, 14), (14, 16)  
]

COLORS_SKELETON = (50, 125, 25) 
COLORS_KEYPOINT = (0, 165, 255)
COLORS_BOX = (0, 255, 0)  
COLORS_REGION = (255, 0, 0)

# Các hàm helper giữ nguyên
def draw_region(frame, points):
    for i in range(len(points)):
        if i < len(points) - 1:
            cv2.line(frame, points[i], points[i + 1], COLORS_REGION, 2)
        else:
            cv2.line(frame, points[i], points[0], COLORS_REGION, 2)
    return frame

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

prev_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = cv2.getTickCount()
    frame = cv2.resize(frame, (1080, 720))
    frame = draw_region(frame, region_points)

    results = model_pose.predict(source=frame, imgsz=640, conf=0.25, verbose=False)

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints.xy.cpu().numpy()

        for box, keypoint in zip(boxes, keypoints):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0] * 100
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if point_in_polygon((center_x, center_y), region_points) and conf > 25:
                # Cắt vùng chứa người
                padding = 35
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding) 
                crop_x2 = min(frame.shape[1], x2 + padding)
                crop_y2 = min(frame.shape[0], y2 + padding)
                
                cropped_image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Phân loại với efficientnet B2
                if cropped_image.size > 0:
                    # Chuyển đổi sang PIL Image
                    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                    # Transform
                    input_tensor = transform(pil_image).unsqueeze(0).to(device)
                    
                    # Dự đoán
                    with torch.no_grad():
                        output = model_classify(input_tensor)
                        _, predicted = torch.max(output, 1)
                        class_name = "not_working" if predicted.item() == 0 else "working"
                
                # Vẽ kết quả phân loại
                cv2.putText(frame, class_name, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS_BOX, 2)

                # Vẽ skeleton
                for i, (x, y) in enumerate(keypoint):
                    if x > 0 and y > 0 and x < frame.shape[1] and y < frame.shape[0]:
                        cv2.circle(frame, (int(x), int(y)), 5, COLORS_KEYPOINT, -1)

                for (p1, p2) in SKELETON_EDGES:
                    x1, y1 = keypoint[p1]
                    x2, y2 = keypoint[p2]
                    if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and 
                        x1 < frame.shape[1] and y1 < frame.shape[0] and 
                        x2 < frame.shape[1] and y2 < frame.shape[0]):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS_SKELETON, 2)

    fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('YOLO11-Pose Skeleton with Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
