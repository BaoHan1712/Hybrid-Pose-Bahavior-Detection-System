import matplotlib.pyplot as plt
import os

# Đọc số lượng ảnh từ thư mục dataset
dataset_dir = "dataset"
labels = ['working', 'not_working']
sizes = []

for label in labels:
    path = os.path.join(dataset_dir, label)
    if os.path.exists(path):
        num_images = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        sizes.append(num_images)

# Màu sắc cho từng phần
colors = ['#99FF99', '#FF9999']

# Tính phần trăm
total = sum(sizes)
percentages = [round(size/total*100, 1) for size in sizes]

# Tạo biểu đồ
plt.figure(figsize=(10, 8))
plt.pie(sizes, 
        labels=[f'{l}\n{p}%' for l, p in zip(labels, percentages)],
        colors=colors,
        autopct='',
        startangle=90)

# Thêm tiêu đề
plt.title('Distribution of Image Counts by Dataset')

# Tạo hình tròn hoàn chỉnh
plt.axis('equal')

# Hiển thị biểu đồ
plt.show()
