import cv2


def draw_region(frame, points):
    # Vẽ các điểm và đường nối
    for i in range(len(points)):
        if i < len(points) - 1:
            cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)
        if i == len(points) - 1 and len(points) == 4:
            cv2.line(frame, points[i], points[0], (0, 255, 0), 2)
        cv2.circle(frame, points[i], 5, (0, 0, 255), -1)
    return frame

def mouse_callback(event, x, y, flags, params):
    points, frame = params
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        # Cập nhật vị trí điểm cuối cùng khi kéo
        if flags & cv2.EVENT_FLAG_LBUTTON and points:
            points[-1] = (x, y)

# Đọc video
cap = cv2.VideoCapture(r"data/v5.mp4")
assert cap.isOpened(), "Error reading video file"

# Đọc frame đầu tiên để vẽ region
ret, frame = cap.read()
if not ret:
    print("Không thể đọc video")
    exit()
frame = cv2.resize(frame, (1080, 720))

# Tạo cửa sổ và thiết lập callback
points = []
cv2.namedWindow("Draw Region")
cv2.setMouseCallback("Draw Region", mouse_callback, (points, frame))

frame_copy = frame.copy()
while True:
    current_frame = frame_copy.copy()
    # Vẽ region
    current_frame = draw_region(current_frame, points)
    
    # Hiển thị frame
    cv2.imshow("Draw Region", current_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Reset points
        points = []
    elif key == ord('c'):  # Confirm and continue
        if len(points) == 4:
            print("Region points:", points)
            break
    elif key == 27:  # ESC để thoát
        break

cv2.destroyAllWindows()
cap.release()
