# Kế hoạch Tạo Slide Chương 7: Chuyên sâu về Keras (A Deep Dive on Keras)

## 1. Phân tích Hiện trạng Chương 7
- **Chủ đề cốt lõi:** Sau khi đã hiểu về quy trình ML cơ bản, chương này đi sâu vào các công cụ mạnh mẽ của Keras để giải quyết các kiến trúc phức tạp: Xây dựng đồ thị đa luồng (Functional API), tùy biến lớp (Model Subclassing), viết các vòng lặp Gradient tùy chỉnh, sử dụng Callbacks và TensorBoard.
- **Tình trạng hình ảnh:** Có 7 hình ảnh (về sơ đồ phân loại vé, kiến trúc Keras, callback, tensorboard). Bắt buộc nhúng đủ 7 ảnh này vào slide.
- **Yêu cầu:** Đạt tối thiểu 15 slides, số hóa hình ảnh thành `Hình 7.X`, và bổ sung chi tiết học thuật cho các API bậc cao của Deep Learning.

## 2. Cấu trúc Nội dung Chi tiết (Dự kiến 16-17 Slides)
- **Phần 1: Triết lý thiết kế của Keras (~3 slides)**
  - Phương châm "Từng bước phơi bày sự phức tạp" (Progressive disclosure of complexity). Hình ảnh: `progressive_disclosure_of_complexity_models`.
  - Vòng lặp tiến bộ (Idea $\rightarrow$ Experiment $\rightarrow$ Evaluate). Hình ảnh: `the_loop_of_progress`.
- **Phần 2: Ba cách xây dựng Mô hình Keras (~6 slides)**
  - Sequential API (Mô hình chuỗi truyền thống).
  - Functional API (Xử lý đa đầu vào, đa đầu ra). Hình ảnh: Mạng nơ-ron đa luồng `ticket_classifier`.
  - Giám sát luồng Tensor trong Functional API. Hình ảnh: `ticket_classifier_with_shapes`.
  - Dễ dàng mở rộng và tái sử dụng mô hình. Hình ảnh: `updated_ticket_classifier`.
  - Model Subclassing (Kế thừa OOP, giành quyền kiểm soát tối đa).
- **Phần 3: Tùy biến Vòng lặp (Custom Training \& Metrics) (~3 slides)**
  - Tự định nghĩa Loss Function (Custom Losses). Công thức toán học của Mean Squared Error.
  - Tự định nghĩa Thước đo (Custom Metrics). Các phương trình cập nhật trạng thái State (Update State).
  - Phá vỡ hàm `fit()`: Tự viết vòng lặp huấn luyện bằng `tf.GradientTape`. (Phương trình lan truyền ngược Backpropagation).
- **Phần 4: Điều khiển Huấn luyện bằng Callbacks (~2 slides)**
  - Cơ chế tự động hóa: `EarlyStopping` và `ModelCheckpoint`.
  - Tự thiết kế Custom Callback để vẽ biểu đồ Loss. Hình ảnh: `loss_history_callback_example`.
- **Phần 5: Giám sát Trực quan bằng TensorBoard (~2 slides)**
  - Theo dõi Metrics, Đồ thị Tính toán (Computation Graph) và Histograms. Hình ảnh: `tensorboard`.
- **Tổng kết (1 slide)**

## 3. Các bước Triển khai
1. **Viết mã LaTeX:** Khởi tạo `TaiLieu/slideDL/Chapter07.tex`. Bổ sung code block hoặc công thức Toán cho `GradientTape` và `Custom Metric`.
2. **Biên dịch:** Chạy `pdflatex` 2 lần liên tiếp.
3. **Hoàn thiện Tasks:** Cập nhật file `.md`.
