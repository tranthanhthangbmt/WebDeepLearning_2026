# Kế hoạch Tạo Slide Chương 18: Kỹ năng Thực chiến \& Tối ưu hóa (Best Practices)

## 1. Phân tích Hiện trạng Chương 18
- **Nội dung cốt lõi:** Chương cuối cùng của giáo trình đóng vai trò là cầu nối từ sinh viên bước ra thực tiễn (Machine Learning Engineer). Nó tập trung vào Tối ưu Siêu tham số, Huấn luyện Phân tán (Multi-GPU/TPU) và Tối ưu phần cứng (Mixed-precision, Quantization).
- **Tình trạng hình ảnh:** Chương này **chỉ có 3 hình ảnh**:
  - `data_and_model_parallelism`: Phân tán Dữ liệu vs Phân tán Mô hình.
  - `floating_pi`: Biểu diễn số dấu phẩy động.
  - `the_loop_of_progress`: Vòng lặp cải tiến liên tục trong AI.
- **Thách thức cực lớn:** Với chỉ 3 hình ảnh, việc tạo ra **tối thiểu 15 slides** chuẩn Đại học đòi hỏi phải đào rất sâu vào \textbf{Toán học} và \textbf{Lý thuyết Khoa học Máy tính}. Tôi sẽ bù đắp lượng hình ảnh ít ỏi bằng các công thức (All-Reduce, Quantization Int8, Ensemble Weights) và phân tích học thuật chuyên sâu.

## 2. Cấu trúc Nội dung Chi tiết (Dự kiến 16-17 Slides)
- **Phần 1: Tối ưu hóa Siêu tham số - Hyperparameter Tuning (4 slides)**
  - Phân biệt Siêu tham số (Hyperparameters) vs Tham số (Weights). Tại sao không thể dùng Gradient Descent để tối ưu Hyperparameter?
  - Các chiến lược tìm kiếm: Lưới (Grid Search), Ngẫu nhiên (Random Search).
  - Tối ưu hóa Bayes (Bayesian Optimization): Sử dụng Surrogate Model để dò tìm không gian siêu tham số một cách thông minh.
  - Công cụ KerasTuner.
- **Phần 2: Suy diễn Đám đông - Model Ensembling (3 slides)**
  - Triết lý Ensemble: "Nhiều mô hình yếu cộng lại thành một mô hình mạnh".
  - Các chiến lược: Bagging, Boosting, và Averaging.
  - Toán học của Ensemble: Tại sao trung bình của nhiều mô hình độc lập luôn tốt hơn 1 mô hình đơn lẻ (Bất đẳng thức Jensen).
- **Phần 3: Huấn luyện Phân tán - Distributed Training (3 slides)**
  - Tốc độ huấn luyện và giới hạn phần cứng.
  - Data Parallelism (Chẻ dữ liệu) vs Model Parallelism (Chẻ mô hình). Hình: `data_and_model_parallelism.1d1087a1.png`.
  - Cơ chế Đồng bộ Gradient (All-Reduce Algorithm) giữa các GPUs.
- **Phần 4: Kỹ thuật Tối ưu Bộ nhớ - Mixed-Precision \& Quantization (4 slides)**
  - Biểu diễn số thực trong bộ nhớ máy tính (IEEE 754). Hình: `floating_pi.b6d4aaaf.png`.
  - Mixed-Precision Training: Dùng FP16 để tăng tốc Tensor Cores, dùng FP32 để lưu Weights tránh Underflow.
  - Lượng tử hóa (Quantization): Ép kiểu từ Float32 xuống Int8. Công thức Quantization: $x_q = \text{round}(\frac{x}{S}) + Z$.
  - Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT).
- **Phần 5: Vòng lặp Tiến bộ \& Tổng kết (2 slides)**
  - Vòng lặp Đánh giá - Phân tích - Tinh chỉnh (The Loop of Progress). Hình: `the_loop_of_progress.4bb26a08.png`.
  - Tổng kết toàn bộ Chương 18.

## 3. Các bước Triển khai
1. **Viết mã LaTeX:** Khởi tạo `TaiLieu/slideDL/Chapter18.tex`. (Chứa nhiều công thức toán bù đắp hình ảnh).
2. **Biên dịch:** Chạy `pdflatex` 2 lần liên tiếp.
3. **Hoàn thiện Tasks:** Báo cáo kết quả và cập nhật hệ thống.
