# Kế hoạch Xây dựng Slide Chương 14: SUY DIỄN XÁC SUẤT THEO THỜI GIAN (Probabilistic Reasoning Over Time)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Tổng hợp và trình bày các kiến thức chuyên sâu về Suy diễn theo thời gian (chuỗi thời gian) từ sách AIMA 4th edition. Tập trung vào HMM, Kalman Filters và DBN.
- **Yêu cầu kỹ thuật LaTeX:** 
  - Đảm bảo biên dịch ổn định trên `pdflatex` với `aima2e-slides.sty`.
  - Công thức Toán học nâng cao: Ma trận chuyển đổi, Phân phối Gaussian (chuẩn), và các tích vô hướng, phép nhân ma trận. Sẽ sử dụng môi trường chuẩn, các chỉ số $t, t+1, 1:t$ phải căn chỉnh mượt mà.

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 14: SUY DIỄN XÁC SUẤT THEO THỜI GIAN
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Liệt kê các danh mục cốt lõi từ 14.1 đến 14.5.

### 14.1 Thời gian và Sự Không Chắc Chắn (Time and Uncertainty)
- Trạng thái và quan sát theo thời gian ($X_t$ và $E_t$).
- Giả định Markov (Markov Assumption): Hiện tại chỉ phụ thuộc vào Quá khứ gần nhất.
- Mô hình chuyển đổi (Transition Model) và Mô hình cảm biến (Sensor Model).
- *Hình ảnh dự kiến:* `figure_14.1.jpg` (Đồ thị mạng Markov).

### 14.2 Suy luận trong các Mô hình Thời gian
- Giới thiệu 4 bài toán cơ bản của Suy luận theo thời gian:
  1. **Lọc (Filtering):** Tính trạng thái hiện tại $\mathbf{P}(X_t | e_{1:t})$.
  2. **Dự đoán (Prediction):** Tính trạng thái tương lai $\mathbf{P}(X_{t+k} | e_{1:t})$.
  3. **Làm mượt (Smoothing):** Đánh giá lại quá khứ $\mathbf{P}(X_k | e_{1:t})$ với $k < t$.
  4. **Tìm chuỗi có khả năng nhất (Most likely sequence):** Viterbi algorithm.
- Đưa ra định lý đệ quy về Filtering (Forward message) và Smoothing (Backward message).

### 14.3 Mô hình Markov ẩn (Hidden Markov Models - HMM)
- HMM là gì? (Một mạng Bayes đặc biệt mà trạng thái là 1 biến rời rạc đơn).
- Các thuật toán ma trận rút gọn (Cập nhật Forward-Backward bằng đại số tuyến tính: $\mathbf{f}_{1:t+1} = \alpha \mathbf{O}_{t+1} \mathbf{T}^T \mathbf{f}_{1:t}$).
- **Ví dụ điển hình:** Bài toán định vị Robot (Localization) trong mê cung.
- *Hình ảnh dự kiến:* Cập nhật xác suất vị trí robot trong bài toán định vị.

### 14.4 Bộ lọc Kalman (Kalman Filters)
- Tại sao cần Kalman Filter? (HMM chỉ áp dụng cho không gian rời rạc, không dùng được cho thế giới vật lý liên tục như radar theo dõi tên lửa, xe tự lái).
- Phân phối Gaussian trong lọc Kalman: Trung bình $\mu$ và Ma trận hiệp phương sai $\Sigma$.
- Ví dụ đơn giản: Chim bay theo đường thẳng (Lọc Kalman một chiều).
- Tính ứng dụng: Mượt mà hóa (Smoothing) các quỹ đạo nhiễu thành quỹ đạo chuẩn xác.

### 14.5 Mạng Bayes động (Dynamic Bayesian Networks - DBN)
- Cấu trúc mạng DBN: Kết hợp ưu điểm của mạng Bayes (biểu diễn nhiều biến) và HMM (theo thời gian).
- Tính toán suy luận chính xác trong DBN: Hiện tượng "Rolled out" và kích thước tham số.
- Suy luận xấp xỉ trong DBN: **Bộ lọc Hạt (Particle Filtering)**.
- Ý tưởng cốt lõi của Particle Filtering: Biểu diễn xác suất bằng một bầy "Hạt" (Particles) và cập nhật qua các bước: Lan truyền (Propagate), Gán trọng số (Weight) và Lấy mẫu lại (Resample).
- *Hình ảnh dự kiến:* Biểu đồ quá trình phân tán và hội tụ của Bộ lọc Hạt.

### Tóm tắt Chương 14
- Nền tảng cốt lõi của công nghệ xe tự hành, nhận dạng giọng nói, và theo dõi mục tiêu quân sự.
- Từ thế giới rời rạc đơn giản (HMM), tiến tới thế giới liên tục (Kalman Filter) và không gian phức tạp đa biến (DBN / Particle Filter).

## 3. Kế hoạch Hiện thực hóa (Thực thi)

1. Tinh chỉnh từ ngữ tiếng Việt: "Transition model" $\rightarrow$ Mô hình chuyển đổi, "Smoothing" $\rightarrow$ Làm mượt, "Particle filtering" $\rightarrow$ Lọc hạt.
2. Thiết kế slide với các block lý thuyết cô đọng `\blob`, công thức HMM và Kalman phải đứng thành từng dòng biệt lập để làm nổi bật sự thanh lịch của toán học học thuật.
3. Kỹ thuật Particle Filter sẽ được thiết kế rất chi tiết, có mô phỏng các bước Propagate - Weight - Resample.
4. Biên dịch thử mã LaTeX để xử lý triệt để bất cứ lỗi cú pháp nào nếu có bằng `pdflatex`.

---
**Ghi chú:** Đây là bản quy hoạch lý thuyết tinh túy nhất cho Chương 14. Thầy kiểm tra nếu thấy kế hoạch này "Đạt" chuẩn của trường thì hãy phản hồi để tôi tiếp tục lập kế hoạch cho **Chương 15 (Ra Quyết Định - Making Simple Decisions)** nhé!
