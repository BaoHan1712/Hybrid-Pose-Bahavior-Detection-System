import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Arrow

# Tạo figure với kích thước lớn hơn
plt.figure(figsize=(12, 5))

# Đọc 2 hình ảnh
img1 = mpimg.imread('ii.jpeg')  # Thay đường dẫn ảnh của bạn
img2 = mpimg.imread('iu.jpg')  # Thay đường dẫn ảnh của bạn

# Tạo 2 subplot
plt.subplot(1, 1, 1)
plt.imshow(img1)
plt.title('Original image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title('Recognized')
plt.axis('off')

# Thêm mũi tên
plt.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.5),
            xycoords='figure fraction',
            arrowprops=dict(arrowstyle='->',
                          color='red',
                          lw=2))

# Điều chỉnh khoảng cách giữa các subplot
plt.tight_layout()

# Hiển thị hình
plt.show()
