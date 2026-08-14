# Kế hoạch Tạo Slide Chương 4: Phân loại và Hồi quy (Classification & Regression)

## 1. Yêu cầu Đặc thù của Chương 4
- **Chủ đề:** Đi sâu vào 3 bài toán kinh điển của Học Máy: Phân loại nhị phân (Binary Classification), Phân loại đa lớp (Multiclass Classification) và Hồi quy vô hướng (Scalar Regression).
- **Yêu cầu độ dài:** Tối thiểu 15 slides (dự kiến 25-30 slides) nhằm đảm bảo chi tiết học thuật, định nghĩa và công thức toán học nền tảng.
- **Yêu cầu hiển thị hình ảnh:** 
  - Tích hợp toàn bộ **11 hình ảnh** trong thư mục `images/ch04/`.
  - Tự động đánh số theo cú pháp "Hình 4.X" thông qua cấu hình `\renewcommand{\thefigure}{4.\arabic{figure}}`.

## 2. Giải pháp Thiết kế (LaTeX Beamer)
- Sử dụng template **Beamer (Theme Madrid, 16:9)** kết hợp với gói `amsmath` để hiển thị tốt công thức toán.
- Thiết kế chia cột linh hoạt (columns) khi cần phân tích trực tiếp biểu đồ hàm mất mát (Loss) và độ chính xác (Accuracy).

## 3. Cấu trúc Nội dung Chi tiết (>15 Slides)
- **Phần 1: Các khái niệm cốt lõi (Glossary - 3 slides)**
  - Phân biệt Sample, Target, Prediction, Loss.
  - Phân biệt các loại Classification (Binary, Multiclass, Multilabel) và Regression.
- **Phần 2: Phân loại Nhị phân (Tập dữ liệu IMDb - 6 slides)**
  - Phân tích bài toán đánh giá cảm xúc (Positive/Negative).
  - Kiến trúc mạng nơ-ron 3 lớp. (Hình: `3_layer_network.cf1b1cd7.png`).
  - Phân tích toán học hàm kích hoạt (Activation Functions): Tại sao cần tính phi tuyến?
    - Relu (Công thức $f(x) = \max(0, x)$ và Hình `The-rectified-linear-unit-function.351095bf.png`).
    - Sigmoid (Công thức $f(x) = \frac{1}{1 + e^{-x}}$ và Hình `The-sigmoid-function.eac1368d.png`).
  - Phân tích hiện tượng Overfitting qua biểu đồ huấn luyện. (Hình `imdb_loss_plot`, `imdb_accuracy_plot`).
- **Phần 3: Phân loại Đa lớp (Tập dữ liệu Reuters - 4 slides)**
  - Bài toán gán nhãn 46 chủ đề tin tức (Single-label, multiclass).
  - So sánh hàm mất mát Binary Crossentropy vs Categorical Crossentropy.
  - Phân tích biểu đồ và khái niệm Top-3 Accuracy. (Hình `reuters_loss_plot`, `reuters_accuracy_plot`, `reuters_top_3_accuracy_plot`).
- **Phần 4: Hồi quy Vô hướng (Dự đoán giá nhà Boston - 5 slides)**
  - Khái niệm liên tục, sử dụng độ đo MAE (Mean Absolute Error).
  - Vấn đề thiếu hụt dữ liệu (Small datasets) và Toán học của K-Fold Cross Validation. (Hình `3-fold-cross-validation.40bb5356.png`).
  - Trích xuất thông tin từ biểu đồ biến động MAE theo epochs. (Hình `california_housing_validation_mae_plot`, `california_housing_validation_mae_plot_zoomed`).

## 4. Các bước Triển khai Thực tế
1. **Khởi tạo và Soạn thảo:** Dùng `write_to_file` tạo mã nguồn LaTeX `TaiLieu/slideDL/Chapter04.tex` với bố cục chặt chẽ theo cấu trúc trên.
2. **Biên dịch:** Chạy lệnh `pdflatex` **2 lần liên tiếp** để tạo trang Mục lục (TOC) hoàn chỉnh và chốt đánh số Hình ảnh.
3. **Hoàn thiện:** Cập nhật tài liệu `task.md` và `walkthrough.md`.
