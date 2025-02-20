import os
import shutil
import random

# Đường dẫn thư mục dữ liệu gốc
data_dir = "dataset"  
output_dir = "dataset_split"  

# Tỷ lệ chia tập dữ liệu
train_ratio = 0.8

# Tạo thư mục output
os.makedirs(output_dir, exist_ok=True)
for subset in ['train', 'test']:
    for class_name in ['working', 'not_working']:
        os.makedirs(os.path.join(output_dir, subset, class_name), exist_ok=True)

# Duyệt qua từng class
for class_name in ['working', 'not_working']:
    class_path = os.path.join(data_dir, class_name)
    images = os.listdir(class_path)
    
    # Xáo trộn dữ liệu
    random.shuffle(images)
    
    # Chia dữ liệu
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    
    # Copy ảnh vào thư mục train
    for img_name in train_images:
        src_path = os.path.join(class_path, img_name)
        dst_path = os.path.join(output_dir, "train", class_name, img_name)
        shutil.copy(src_path, dst_path)
    
    # Copy ảnh vào thư mục test  
    for img_name in test_images:
        src_path = os.path.join(class_path, img_name)
        dst_path = os.path.join(output_dir, "test", class_name, img_name)
        shutil.copy(src_path, dst_path)

print("Chia dữ liệu hoàn tất!")
