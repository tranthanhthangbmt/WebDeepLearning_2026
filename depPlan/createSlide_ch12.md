# Kế hoạch Tạo Slide Chương 12: Phát hiện Vật thể (Object Detection)

## 1. Phân tích Hiện trạng Chương 12
- **Nội dung cốt lõi:** Làm rõ sự khác biệt giữa Object Detection và Segmentation. Khám phá 2 trường phái kiến trúc chính: Two-stage (R-CNN family) ưu tiên độ chính xác và Single-stage (YOLO, RetinaNet) ưu tiên tốc độ (Real-time).
- **Tình trạng hình ảnh:** Có **9 hình ảnh** bao phủ từ luồng xử lý R-CNN, sơ đồ Feature Pyramid Network (FPN), cách chia Grid của YOLO và quá trình lọc nhiễu Non-Maximum Suppression (NMS).
- **Thách thức:** Với 9 hình ảnh, để đạt độ sâu học thuật và số lượng tối thiểu 15 slides, cần đi sâu vào Toán học của Thuật toán **YOLO (You Only Look Once)** bao gồm: Hồi quy Box $(x, y, w, h)$, tính toán **IoU (Intersection over Union)** và cơ chế hoạt động của thuật toán **NMS**.

## 2. Cấu trúc Nội dung Chi tiết (Dự kiến 18-20 Slides)
- **Phần 1: Giới thiệu Bài toán (3 slides)**
  - Tầm quan trọng của Phát hiện vật thể (Đếm, Theo dõi, Trích xuất vùng ảnh). Khác biệt so với Segmentation. Hình ảnh: `object-detection.7c5cbfd4.png`.
  - Bộ dữ liệu COCO (Common Objects in Context) - Chuẩn vàng của Detection. Hình ảnh: `coco-example.c43c39d6.png`.
- **Phần 2: Kiến trúc Hai giai đoạn (Two-stage Detectors) (4 slides)**
  - Cơ chế Region Proposals (Vùng đề xuất). 
  - Quy trình xử lý R-CNN: Đề xuất vùng $\rightarrow$ Đưa qua CNN $\rightarrow$ Phân loại \& Tinh chỉnh Box. Hình ảnh: `r-cnn-pipeline.8fe83666.png`.
  - Nhược điểm: Nặng nề, không thể chạy Real-time trên xe tự lái.
- **Phần 3: Kiến trúc Một giai đoạn (Single-stage Detectors) (4 slides)**
  - RetinaNet và giải pháp Feature Pyramid Network (FPN) cho vật thể kích thước siêu nhỏ. Hình ảnh: `feature-pyramid-network.83b6f108.png`.
  - Trực quan hóa kết quả đầu ra của Single-stage. Hình ảnh: `retinanet-output.0a67b6e8.png`.
- **Phần 4: Thuật toán YOLO (You Only Look Once) (7 slides)**
  - Kỷ nguyên mới: Chuyển Detection thành bài toán Hồi quy (Regression) trên Lưới (Grid). 
  - Cách chia Lưới $S \times S$ và tạo Anchor Boxes. Hình ảnh: `yolo-diagram.b7347ca6.png` và `yolo-targets.7b3c7aed.png`.
  - Toán học Hồi quy Tọa độ hộp: Dự đoán Vector $[p_c, b_x, b_y, b_w, b_h, c_1, c_2...]$.
  - Hàng ngàn hộp nhiễu được tạo ra. Hình ảnh: `yolo-predictions-all.de18d520.png`.
  - Toán học của bộ lọc: **IoU (Intersection over Union)** và **Non-Maximum Suppression (NMS)**.
  - Kết quả làm sạch và trích xuất cuối cùng. Hình ảnh: `yolo-predictions.d621592d.png`.
- **Tổng kết (1 slide)**

## 3. Các bước Triển khai
1. **Viết mã LaTeX:** Khởi tạo `TaiLieu/slideDL/Chapter12.tex`.
2. **Biên dịch:** Chạy `pdflatex` 2 lần liên tiếp.
3. **Hoàn thiện Tasks:** Báo cáo kết quả và cập nhật hệ thống.
