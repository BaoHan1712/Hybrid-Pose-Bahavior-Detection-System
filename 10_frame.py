import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import tkinter as tk
from tkinter import ttk, filedialog
import os
from datetime import datetime
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import time
import io
import openpyxl



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


class BehaviorAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Phân Tích Hành Vi")
        
        # Khởi tạo các biến
        self.video_path = None
        self.is_analyzing = False
        self.total_frames = 0
        self.working_frames = 0
        self.not_working_frames = 0
        self.person_count = 0
        self.last_save_time = time.time()
        
        # Load models
        self.load_models()
        
        # Thêm vào phần khởi tạo hiện có
        self.analysis_data = {
            'timestamp': [],
            'working_percent': [],
            'not_working_percent': [],
            'person_count': []
        }
        
        # Tạo GUI
        self.create_gui()
        
    def load_models(self):
        self.model_pose = YOLO("model/yolo11m-pose.engine")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_classify = EfficientNetB2_Model(num_classes=2).to(self.device)
        self.model_classify.load_state_dict(torch.load(r"model/best_EfficientNetB2.pth", map_location=self.device))
        self.model_classify.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

    def create_gui(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Nút chọn video
        ttk.Button(main_frame, text="Chọn Video", command=self.select_video).grid(row=0, column=0, pady=5)
        
        # Nút bắt đầu/dừng phân tích
        self.analyze_btn = ttk.Button(main_frame, text="Bắt Đầu Phân Tích", command=self.toggle_analysis)
        self.analyze_btn.grid(row=0, column=1, pady=5)
        
        # Frame hiển thị video
        self.video_frame = ttk.Label(main_frame)
        self.video_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Frame hiển thị thống kê
        stats_frame = ttk.LabelFrame(main_frame, text="Thống Kê", padding="5")
        stats_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Labels thống kê
        self.working_label = ttk.Label(stats_frame, text="Đang làm việc: 0%")
        self.working_label.grid(row=0, column=0, padx=5)
        
        self.not_working_label = ttk.Label(stats_frame, text="Không làm việc: 0%")
        self.not_working_label.grid(row=0, column=1, padx=5)
        
        self.person_count_label = ttk.Label(stats_frame, text="Số người: 0")
        self.person_count_label.grid(row=1, column=0, columnspan=2, pady=5)

        # Thêm frame cho các nút điều khiển phụ
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Thêm nút xem biểu đồ
        ttk.Button(control_frame, text="Xem Biểu Đồ", 
                  command=self.show_analytics).grid(row=0, column=0, padx=5)
        
        # Thêm nút xuất báo cáo
        ttk.Button(control_frame, text="Xuất Báo Cáo", 
                  command=self.export_report).grid(row=0, column=1, padx=5)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.show_frame()

    def toggle_analysis(self):
        if not self.is_analyzing:
            self.is_analyzing = True
            self.analyze_btn.configure(text="Dừng Phân Tích")
            self.start_analysis()
        else:
            self.is_analyzing = False
            self.analyze_btn.configure(text="Bắt Đầu Phân Tích")

    def start_analysis(self):
        if not self.video_path:
            return
            
        # Tạo thư mục export nếu chưa tồn tại
        if not os.path.exists('export'):
            os.makedirs('export')
        
        # Tạo writer cho video output
        output_path = f"export/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        self.analyze_frame()

    def analyze_frame(self):
        if not self.is_analyzing:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.out.release()
            return

        # Phân tích frame
        results = self.model_pose.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
        
        working_count = 0
        not_working_count = 0
        person_count = 0

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints.xy.cpu().numpy()
            
            for box, keypoint in zip(boxes, keypoints):
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Cắt và phân loại
                cropped_image = frame[y1:y2, x1:x2]
                if cropped_image.size > 0:
                    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model_classify(input_tensor)
                        _, predicted = torch.max(output, 1)
                        label = "Working" if predicted.item() == 1 else "Not Working"
                        color = (0, 255, 0) if predicted.item() == 1 else (0, 0, 255)
                        if predicted.item() == 1:
                            working_count += 1
                        else:
                            not_working_count += 1

                # Vẽ bbox với nhãn
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Thêm background cho text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, 
                            (x1, y1 - text_size[1] - 10), 
                            (x1 + text_size[0], y1),
                            color, -1)
                            
                # Vẽ text
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Vẽ skeleton
                self.draw_skeleton(frame, keypoint)

        # Cập nhật thống kê
        total = working_count + not_working_count
        if total > 0:
            working_percent = (working_count / total) * 100
            not_working_percent = (not_working_count / total) * 100
            
            # Hiển thị thống kê trên frame
            cv2.putText(frame, f"Working: {working_percent:.1f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Not Working: {not_working_percent:.1f}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"People: {person_count}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Cập nhật GUI
            self.working_label.configure(text=f"Đang làm việc: {working_percent:.1f}%")
            self.not_working_label.configure(text=f"Không làm việc: {not_working_percent:.1f}%")
            self.person_count_label.configure(text=f"Số người: {person_count}")

            # Thêm vào sau phần cập nhật GUI
            current_time = time.time()
            if current_time - self.last_save_time >= 1.0:
                if total > 0:
                    self.update_analysis_data(working_percent, not_working_percent, person_count)
                    self.last_save_time = current_time

        # Ghi frame đã xử lý
        self.out.write(frame)
        
        # Hiển thị frame
        self.show_frame(frame)
        
        # Tiếp tục vòng lặp
        self.root.after(30, self.analyze_frame)

    def draw_skeleton(self, frame, keypoint):
        SKELETON_EDGES = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        for i, (x, y) in enumerate(keypoint):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 165, 255), -1)
                
        for (p1, p2) in SKELETON_EDGES:
            x1, y1 = keypoint[p1]
            x2, y2 = keypoint[p2]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (50, 125, 25), 2)

    def show_frame(self, frame=None):
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 600))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_frame.configure(image=photo)
        self.video_frame.image = photo

    def update_analysis_data(self, working_percent, not_working_percent, person_count):
        self.analysis_data['timestamp'].append(datetime.now())
        self.analysis_data['working_percent'].append(working_percent)
        self.analysis_data['not_working_percent'].append(not_working_percent)
        self.analysis_data['person_count'].append(person_count)

    def show_analytics(self):
        # Tạo cửa sổ mới cho biểu đồ
        analytics_window = tk.Toplevel(self.root)
        analytics_window.title("Phân Tích Dữ Liệu")
        
        # Tạo figure với 3 subplot
        fig = plt.Figure(figsize=(12, 8))
        
        # Biểu đồ phần trăm làm việc theo thời gian
        ax1 = fig.add_subplot(221)
        ax1.plot(self.analysis_data['timestamp'], 
                self.analysis_data['working_percent'], 
                label='Làm việc', color='green')
        ax1.plot(self.analysis_data['timestamp'], 
                self.analysis_data['not_working_percent'], 
                label='Không làm việc', color='red')
        ax1.set_title('Tỷ lệ làm việc theo thời gian')
        ax1.legend()
        
        # Biểu đồ số người theo thời gian
        ax2 = fig.add_subplot(222)
        ax2.plot(self.analysis_data['timestamp'], 
                self.analysis_data['person_count'], 
                color='blue')
        ax2.set_title('Số người theo thời gian')
        
        # Biểu đồ phân phối trạng thái
        ax3 = fig.add_subplot(223)
        working_avg = np.mean(self.analysis_data['working_percent'])
        not_working_avg = np.mean(self.analysis_data['not_working_percent'])
        ax3.pie([working_avg, not_working_avg], 
                labels=['Làm việc', 'Không làm việc'],
                colors=['green', 'red'],
                autopct='%1.1f%%')
        ax3.set_title('Phân phối trạng thái trung bình')
        
        # Thêm canvas vào cửa sổ
        canvas = FigureCanvasTkAgg(fig, analytics_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_report(self):
        try:
            # Tạo DataFrame từ dữ liệu phân tích
            df = pd.DataFrame({
                'Thời gian': self.analysis_data['timestamp'],
                'Tỷ lệ làm việc (%)': self.analysis_data['working_percent'],
                'Tỷ lệ không làm việc (%)': self.analysis_data['not_working_percent'],
                'Số người': self.analysis_data['person_count']
            })
            
            # Tạo thư mục reports nếu chưa tồn tại
            if not os.path.exists('reports'):
                os.makedirs('reports')
            
            # Tạo tên file với timestamp
            filename = os.path.join('reports', f"bao_cao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            
            # Tạo các biểu đồ
            fig = plt.Figure(figsize=(12, 8))
            
            # Biểu đồ phần trăm làm việc theo thời gian
            ax1 = fig.add_subplot(221)
            ax1.plot(self.analysis_data['timestamp'], 
                    self.analysis_data['working_percent'], 
                    label='Làm việc', color='green')
            ax1.plot(self.analysis_data['timestamp'], 
                    self.analysis_data['not_working_percent'], 
                    label='Không làm việc', color='red')
            ax1.set_title('Tỷ lệ làm việc theo thời gian')
            ax1.legend()
            
            # Biểu đồ số người theo thời gian
            ax2 = fig.add_subplot(222)
            ax2.plot(self.analysis_data['timestamp'], 
                    self.analysis_data['person_count'], 
                    color='blue')
            ax2.set_title('Số người theo thời gian')
            
            # Biểu đồ phân phối trạng thái
            ax3 = fig.add_subplot(223)
            working_avg = np.mean(self.analysis_data['working_percent'])
            not_working_avg = np.mean(self.analysis_data['not_working_percent'])
            ax3.pie([working_avg, not_working_avg], 
                    labels=['Làm việc', 'Không làm việc'],
                    colors=['green', 'red'],
                    autopct='%1.1f%%')
            ax3.set_title('Phân phối trạng thái trung bình')

            # Lưu biểu đồ vào buffer
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            plt.close(fig)
            
            # Tạo ExcelWriter với engine='openpyxl'
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet dữ liệu chi tiết
                df.to_excel(writer, sheet_name='Chi tiết', index=False)
                
                # Sheet tổng quan
                summary = pd.DataFrame({
                    'Chỉ số': [
                        'Tỷ lệ làm việc trung bình (%)',
                        'Tỷ lệ không làm việc trung bình (%)',
                        'Số người trung bình',
                        'Thời gian bắt đầu',
                        'Thời gian kết thúc',
                        'Tổng số frame phân tích'
                    ],
                    'Giá trị': [
                        f"{np.mean(self.analysis_data['working_percent']):.2f}",
                        f"{np.mean(self.analysis_data['not_working_percent']):.2f}",
                        f"{np.mean(self.analysis_data['person_count']):.1f}",
                        self.analysis_data['timestamp'][0].strftime('%Y-%m-%d %H:%M:%S'),
                        self.analysis_data['timestamp'][-1].strftime('%Y-%m-%d %H:%M:%S'),
                        len(self.analysis_data['timestamp'])
                    ]
                })
                summary.to_excel(writer, sheet_name='Tổng quan', index=False)

                # Thêm sheet biểu đồ
                worksheet = writer.book.create_sheet('Biểu đồ')
                img = openpyxl.drawing.image.Image(img_buf)
                worksheet.add_image(img, 'A1')
            
            tk.messagebox.showinfo("Thành công", f"Đã xuất báo cáo: {filename}")
            
        except Exception as e:
            tk.messagebox.showerror("Lỗi", f"Không thể xuất báo cáo: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BehaviorAnalysisApp(root)
    root.mainloop()
