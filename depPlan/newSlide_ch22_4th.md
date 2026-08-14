# Kế hoạch Xây dựng Slide Chương 22: HỌC SÂU (Deep Learning)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Sinh viên hiểu cấu trúc, nguyên lý hoạt động, và sức mạnh bùng nổ của mạng Nơ-ron nhân tạo (Deep Learning) bao gồm: Feedforward, CNN, RNN, LSTM và Transfer Learning.
- **Yêu cầu kỹ thuật LaTeX:** 
  - Biên dịch chuẩn bằng `pdflatex`.
  - Môi trường toán học sẽ được sử dụng mạnh để biểu diễn Phép tính Tensor, Gradient (Lan truyền ngược), và các hàm kích hoạt (Activation functions) như ReLU, Sigmoid.

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 22: HỌC SÂU
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Liệt kê các đề mục cốt lõi từ 22.1 đến 22.8.

### 22.1 Các Mạng Truyền Thẳng Đơn Giản (Simple Feedforward Networks)
- Nơ-ron nhân tạo là gì? Cấu trúc của một Perceptron.
- Mạng như những hàm phức tạp: Tại sao cần hàm kích hoạt phi tuyến (Non-linear Activation) như ReLU, Tanh?
- Các gradient và quá trình học.

### 22.2 Các Đồ Thị Tính Toán Cho Học Sâu (Computation Graphs for Deep Learning)
- Mã hóa đầu vào (Input encoding).
- Các tầng đầu ra và các hàm mất mát (Ví dụ: Softmax và Cross-Entropy cho Classification).
- Kiến trúc các tầng ẩn (Hidden layers): Định nghĩa chiều sâu (Deep) của mạng.

### 22.3 Các Mạng Tích Chập (Convolutional Networks - CNN)
- Động lực: Xử lý dữ liệu ảnh quá khổng lồ so với Feedforward. Cấu trúc không gian (Spatial structure).
- Bộ lọc (Filters) và các phép toán Tensor trong mạng CNN.
- Gộp và giảm mẫu (Pooling / Max-pooling).
- Mạng thặng dư (Residual networks - ResNet) để giải quyết lỗi triệt tiêu Gradient (Vanishing Gradient).

### 22.4 Các Thuật Toán Học (Learning Algorithms)
- **Lan truyền ngược (Backpropagation):** Tính toán các gradient trong các đồ thị tính toán bằng Quy tắc dây chuyền (Chain Rule).
- Chuẩn hóa theo lô (Batch normalization) để mạng học ổn định và nhanh hơn.

### 22.5 Khái Quát Hóa (Generalization)
- Chọn lựa kiến trúc mạng và Tìm kiếm kiến trúc mạng nơ-ron (Neural architecture search - NAS).
- Tránh khớp quá mức (Overfitting): Phân rã trọng số (Weight decay / $L_2$ regularization) và **Dropout**.

### 22.6 Các Mạng Nơ-ron Hồi Quy (Recurrent Neural Networks - RNN)
- Xử lý dữ liệu chuỗi (Thời gian, Âm thanh, Văn bản).
- Bộ nhớ ngắn hạn dài (Long short-term memory RNNs - LSTM): Cơ chế các Cổng (Gates) để lưu trữ thông tin dài hạn.

### 22.7 Học Không Giám Sát và Học Chuyển Giao (Unsupervised / Transfer Learning)
- Tự mã hóa (Autoencoders) và sinh dữ liệu (GANs).
- Học chuyển giao (Transfer learning): "Đứng trên vai người khổng lồ", tái sử dụng trọng số từ mạng đã huấn luyện (như ResNet-50) cho tác vụ mới.

### 22.8 Các Ứng Dụng (Applications)
- Thị giác máy tính (Vision), Xử lý ngôn ngữ tự nhiên (NLP) và Học tăng cường sâu (Deep Reinforcement Learning - DQN, AlphaGo).

### Tóm tắt Chương 22
- Học sâu đã cách mạng hóa toàn bộ ngành Trí tuệ nhân tạo. Sức mạnh của nó đến từ Dữ liệu khổng lồ, Năng lực tính toán (GPU) và sự lặp lại đơn giản của các tầng phi tuyến.

## 3. Kế hoạch Hiện thực hóa (Thực thi)

1. Tinh gọn lý thuyết: Viết thẳng vào nguyên lý của CNN (Tích chập, Gộp) và RNN (Hồi quy, Chuỗi), tránh lan man dông dài chữ.
2. Trọng tâm hình ảnh: Chương này bắt buộc phải tràn ngập hình ảnh mô phỏng kiến trúc mạng (Hình ảnh 3D của Tensor, Filter trong CNN). Sẽ chèn các figure từ bộ AIMA gốc một cách nghệ thuật.
3. Xử lý thuật ngữ LaTeX Toán: Các công thức Backpropagation và chuỗi đạo hàm $\frac{\partial L}{\partial w}$ cần thiết kế trên khối `block` riêng để sinh viên không bị rối mắt.
4. Biên dịch thử với `pdflatex` trước khi tiến hành nghiệm thu.

---
**Ghi chú:** Kế hoạch Chương 22 đã được hoàn thành. Thầy hãy kiểm duyệt các bản kế hoạch (từ 20 đến 22). Nếu đạt chuẩn, thầy cho lệnh, tôi sẽ bắt đầu lên kế hoạch cho các chương từ **23** trở đi, hoặc bắt đầu triển khai code slide nhé!
