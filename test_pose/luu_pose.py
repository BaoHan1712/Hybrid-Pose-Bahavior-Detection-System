import cv2
import os
from ultralytics import YOLO
import time

# Tạo thư mục save_image nếu chưa tồn tại
if not os.path.exists('save_image'):
    os.makedirs('save_image')

# Biến đếm số ảnh đã lưu
image_counter = 0

# Thêm biến theo dõi thời gian
last_save_time = time.time()

model = YOLO("model\yolo11m-pose.engine")

classnames = ['human']

region_points =[(339, 711), (945, 737), (962, 259), (576, 331)]
cap = cv2.VideoCapture(r"data\v4.mp4")
prev_frame_time = 0

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

def draw_region(frame, points):
    # Vẽ vùng quan sát
    for i in range(len(points)):
        if i < len(points) - 1:
            cv2.line(frame, points[i], points[i + 1], COLORS_REGION, 1)
        else:
            cv2.line(frame, points[i], points[0], COLORS_REGION, 1)
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



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = cv2.getTickCount()
    frame = cv2.resize(frame, (1080, 720))

    frame = draw_region(frame, region_points)

    results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints.xy.cpu().numpy()

        # Tạo danh sách để lưu các ảnh cắt
        cropped_images = []

        for box, keypoint in zip(boxes, keypoints):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0] * 100
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if point_in_polygon((center_x, center_y), region_points) and conf > 30:
                # cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS_BOX, 1)

                # Vẽ các khớp xương
                for i, (x, y) in enumerate(keypoint):
                    if x > 0 and y > 0 and x < frame.shape[1] and y < frame.shape[0]:
                        cv2.circle(frame, (int(x), int(y)), 3, COLORS_KEYPOINT, -1)
                        # cv2.putText(frame, str(i), (int(x), int(y) - 10),
                                #   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS_KEYPOINT, 1)

                # Vẽ đường nối giữa các khớp
                for (p1, p2) in SKELETON_EDGES:
                    x1, y1 = keypoint[p1]
                    x2, y2 = keypoint[p2]
                    if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and 
                        x1 < frame.shape[1] and y1 < frame.shape[0] and 
                        x2 < frame.shape[1] and y2 < frame.shape[0]):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS_SKELETON, 1)

                # Tìm tọa độ min và max của keypoints
                min_x = float('inf')
                min_y = float('inf')
                max_x = float('-inf')
                max_y = float('-inf')
                valid_points = False

                for x, y in keypoint:
                    if x > 0 and y > 0:
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                        valid_points = True

                if valid_points:
                    padding = 35
                    crop_x1 = max(0, int(min_x - padding))
                    crop_y1 = max(0, int(min_y - padding))
                    crop_x2 = min(frame.shape[1], int(max_x + padding))
                    crop_y2 = min(frame.shape[0], int(max_y + padding))
                    
                    cropped_image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if cropped_image.size > 0:
                        cropped_images.append(cropped_image)

        # Lưu tất cả các ảnh đã cắt sau mỗi 0.6 giây
        current_time = time.time()
        if current_time - last_save_time >= 0.6 and cropped_images:
            for idx, img in enumerate(cropped_images):
                image_path = f'save_image/pose_{image_counter}_{idx}.jpg'
                cv2.imwrite(image_path, img)
            image_counter += 1
            last_save_time = current_time

    fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Pose Skeleton', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
