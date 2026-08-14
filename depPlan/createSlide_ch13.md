# Kế hoạch Tạo Slide Chương 13: Dự báo Chuỗi thời gian (Timeseries Forecasting)

## 1. Phân tích Hiện trạng Chương 13
- **Nội dung cốt lõi:** Khám phá đặc tính của dữ liệu dạng chuỗi (Sequence/Timeseries). Giới thiệu Mạng Nơ-ron Truyền hồi (RNN) và đặc biệt là kiến trúc **LSTM (Long Short-Term Memory)** - tiêu chuẩn vàng cho dữ liệu có tính thứ tự. Trình bày các kỹ thuật nâng cao: Recurrent Dropout, Stacked RNN, và Bidirectional RNN.
- **Tình trạng hình ảnh:** Có **12 hình ảnh**. Bao quát từ biểu đồ nhiệt độ (Jena dataset), sơ đồ kiến trúc SimpleRNN, cấu trúc 3 phần chi tiết của LSTM (unrolled), cho đến các biểu đồ so sánh Loss của hàng loạt biến thể mô hình (Dense, Conv1D, LSTM, GRU).
- **Thách thức:** Để trình bày cấu trúc LSTM một cách hàn lâm và dễ hiểu, cần liên kết 3 hình ảnh `unrolled_lstm_1, 2, 3` với **hệ phương trình toán học** của 3 Cổng (Gates): Forget Gate, Input Gate, và Output Gate. Điều này sẽ giúp sinh viên hiểu rõ tại sao LSTM giải quyết được vấn đề Triệt tiêu đạo hàm (Vanishing Gradient).

## 2. Cấu trúc Nội dung Chi tiết (Dự kiến 20 Slides)
- **Phần 1: Giới thiệu Chuỗi thời gian và Dữ liệu (3 slides)**
  - Tầm quan trọng của Yếu tố "Thứ tự" (Temporal Order). Bài toán dự báo (Forecasting).
  - Trực quan hóa Dữ liệu thời tiết Jena (Nhiều năm vs Vài ngày). Hình ảnh: `temperature_over_several_years.365f2e2e.png` và `temperature_over_several_days.975eb51a.png`.
- **Phần 2: Sự thất bại của các Mạng truyền thống (3 slides)**
  - Baseline model sử dụng Dense Network. Hình ảnh: `dense_model_metrics.8448f47a.png`.
  - Baseline model sử dụng 1D Convolution. Hình ảnh: `conv_model_metrics.fe487977.png`.
  - Phân tích tại sao chúng thất bại: Không duy trì được trạng thái (Memory) qua các bước thời gian.
- **Phần 3: Mạng Nơ-ron Truyền hồi - RNN (3 slides)**
  - Cơ chế vòng lặp và cập nhật Trạng thái (State). Hình ảnh: `simplernn.822d53ed.png`.
  - Vấn đề chí mạng: Triệt tiêu đạo hàm (Vanishing Gradients) khi học chuỗi dài.
- **Phần 4: Kiến trúc LSTM - Long Short-Term Memory (7 slides)**
  - Cứu cánh của RNN: Bổ sung Trạng thái Tế bào (Cell State $C_t$) - Đường ray cao tốc.
  - Phân tích cấu trúc LSTM (Phần 1): Trạng thái mang theo. Hình ảnh: `unrolled_lstm_1.d9bee30c.png`.
  - Phân tích cấu trúc LSTM (Phần 2): Cổng Quên (Forget Gate) \& Cổng Đầu vào (Input Gate). Hình ảnh: `unrolled_lstm_2.4145ecdf.png`.
  - Phân tích cấu trúc LSTM (Phần 3): Cổng Đầu ra (Output Gate). Hình ảnh: `unrolled_lstm_3.1f68b33f.png`.
  - Kết quả huấn luyện LSTM cơ bản. Hình ảnh: `lstm_model_metrics.ae01dd09.png`.
- **Phần 5: Các Kỹ thuật Nâng cao (3 slides)**
  - Chống Overfitting với Recurrent Dropout. Hình ảnh: `lstm_dropout_model_metrics.a624dc88.png`.
  - Tăng cường Sức mạnh với Xếp chồng (Stacked GRU). Hình ảnh: `stacked_gru_dropout_model_metrics.5bfbf251.png`.
  - Nhìn về tương lai với Bidirectional RNN. Hình ảnh: `bidirectional_rnn.a38aaba4.png`.
- **Tổng kết (1 slide)**

## 3. Các bước Triển khai
1. **Viết mã LaTeX:** Khởi tạo `TaiLieu/slideDL/Chapter13.tex`. Đưa công thức 3 cổng của LSTM vào một cách trực quan.
2. **Biên dịch:** Chạy `pdflatex` 2 lần liên tiếp.
3. **Hoàn thiện Tasks:** Báo cáo kết quả và cập nhật hệ thống.
