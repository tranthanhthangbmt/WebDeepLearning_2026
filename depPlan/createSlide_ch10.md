# Kế hoạch Tạo Slide Chương 10: Diễn dịch và Trực quan hóa Mạng Tích chập

## 1. Phân tích Hiện trạng Chương 10
- **Nội dung cốt lõi:** Giải quyết định kiến "Deep Learning là Hộp đen (Black box)". Giới thiệu 3 kỹ thuật nội soi mạng CNN: Trực quan hóa Kích hoạt trung gian (Intermediate Activations), Trực quan hóa Bộ lọc (Filter Visualization) bằng Gradient Ascent, và Bản đồ Nhiệt Grad-CAM.
- **Tình trạng hình ảnh:** Có **9 hình ảnh** xuất sắc mô tả quá trình Mạng nơ-ron "bóc tách" một bức ảnh con mèo, các họa tiết kỳ ảo mà Bộ lọc học được, và Bản đồ nhiệt chỉ đích danh "Cái vòi voi".
- **Thách thức:** Cần lồng ghép các phương trình Đạo hàm lên Pixel (Gradient Ascent in Input Space) và Toán học của Grad-CAM để duy trì hàm lượng học thuật cao.

## 2. Cấu trúc Nội dung Chi tiết (Dự kiến 18-20 Slides)
- **Phần 1: Vấn đề "Hộp Đen" trong Deep Learning (2 slides)**
  - Tầm quan trọng của tính Khả diễn dịch (Interpretability) trong AI Y tế / Xe tự lái.
  - Tại sao CNN lại không phải là Hộp đen (Biểu diễn thị giác rõ ràng).
- **Phần 2: Trực quan hóa Biểu diễn Trung gian (5 slides)**
  - Định nghĩa: Kích hoạt (Activations) hay Bản đồ đặc trưng (Feature Maps).
  - Đưa ảnh con mèo vào mạng. Hình ảnh: `cat.135f6a4b.png`.
  - Kích hoạt của 1 Kênh cụ thể (Edge detector). Hình ảnh: `single_filter.8d2772d6.png` và `fifth_activation.3dac2691.png`.
  - Trực quan toàn bộ 128/256 kênh của các Lớp sâu. Hình ảnh: `all_activations.7a8d82cd.png`.
  - Kết luận: Sự chuyển đổi từ "Chi tiết Không gian" (Lớp nông) sang "Khái niệm Trừu tượng" (Lớp sâu).
- **Phần 3: Trực quan hóa Bộ lọc bằng Gradient Ascent (5 slides)**
  - Khái niệm: Đi tìm Bức ảnh hoàn hảo nhất có thể kích hoạt tối đa một Bộ lọc (Filter) cụ thể.
  - Toán học Gradient Ascent trên Pixel: $x_{new} = x_{old} + \alpha \cdot \nabla_x A(x)$. Trái ngược hoàn toàn với việc cập nhật Trọng số!
  - Hình ảnh các họa tiết kết cấu được CNN học (Gợn sóng, Vảy cá, Chấm bi). Hình ảnh: `allfilters.8d050d97.png` và `bicycles.c7a8501c.png`.
- **Phần 4: Bản đồ Kích hoạt Lớp - Grad-CAM (6 slides)**
  - Câu hỏi: "Tại sao AI lại bảo đây là con Voi châu Phi?"
  - Ảnh đầu vào con voi. Hình ảnh: `elephant.6abc731a.jpg`.
  - Nền tảng Toán học Grad-CAM: Lấy đạo hàm của Xác suất Lớp mục tiêu $y^c$ theo Bản đồ đặc trưng $A^k$. Tính Global Average Pooling để ra trọng số $\alpha_k^c$.
  - Công thức Heatmap: $L_{Grad-CAM} = ReLU\left(\sum_k \alpha_k^c A^k\right)$.
  - Hiển thị Bản đồ nhiệt (Heatmap). Hình ảnh: `cam.b66fff28.png`.
  - Chồng Bản đồ nhiệt lên ảnh gốc (Superimposed). Hình ảnh: `elephant_cam.73b7f8e0.jpg`. Rõ ràng AI đã nhìn vào "cái vòi voi".
- **Tổng kết (1 slide)**

## 3. Các bước Triển khai
1. **Viết mã LaTeX:** Khởi tạo `TaiLieu/slideDL/Chapter10.tex`.
2. **Biên dịch:** Chạy `pdflatex` 2 lần liên tiếp.
3. **Hoàn thiện Tasks:** Báo cáo kết quả và cập nhật hệ thống.
