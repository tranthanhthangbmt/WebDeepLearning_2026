# Kế hoạch Tạo Slide Chương 15: Mô hình Ngôn ngữ \& Kiến trúc Transformer

## 1. Phân tích Hiện trạng Chương 15
- **Nội dung cốt lõi:** Đây là chương đỉnh cao nhất của học phần. Bắt đầu từ Mô hình chuỗi (Seq2Seq) dùng cho Dịch máy, đi tới cuộc cách mạng **Attention Mechanism** (Cơ chế chú ý), và kết thúc bằng kiến trúc **Transformer** (Nền tảng của ChatGPT hiện đại).
- **Tình trạng hình ảnh:** Có **8 hình ảnh** cực kỳ quan trọng, bao trùm toàn bộ sự phát triển của NLP hiện đại:
  - `seq2seq-learning` và `seq2seq-rnn`: Kiến trúc Seq2Seq cơ bản.
  - `attention-concept`, `attention`, `attention-scores`: Khái niệm và Toán học của Cơ chế Attention.
  - `query-key-value` và `multi-head-attention`: Đột phá Toán học Self-Attention (Q-K-V).
  - `encoder-decoder`: Sơ đồ toàn cảnh của mạng Transformer.
- **Thách thức:** Phương trình Scaled Dot-Product Attention ($softmax(\frac{QK^T}{\sqrt{d_k}})V$) là công thức quan trọng nhất thập kỷ của AI. Cần phân bổ slide để giải thích cặn kẽ 3 khái niệm Query (Truy vấn), Key (Chìa khóa), và Value (Giá trị).

## 2. Cấu trúc Nội dung Chi tiết (Dự kiến 18-20 Slides)
- **Phần 1: Mô hình Ngôn ngữ \& Seq2Seq (3 slides)**
  - Language Model là gì? Dự đoán từ tiếp theo (Next-token prediction).
  - Bài toán Dịch máy (Machine Translation) và mạng Sequence-to-Sequence. Hình: `seq2seq-learning.0e1e1c31.png`.
  - Điểm yếu "Nút thắt cổ chai" (Bottleneck) của RNN/LSTM. Hình: `seq2seq-rnn.ec377d3b.png`.
- **Phần 2: Cơ chế Chú ý - Attention Mechanism (5 slides)**
  - Sự ra đời của Attention: Cho phép Decoder "nhìn" lại toàn bộ chuỗi Encoder. Hình: `attention-concept.fde57742.png`.
  - Alignment Scores (Điểm chú ý): Làm thế nào máy tính biết nên tập trung vào từ nào? Hình: `attention-scores.2932e0ff.png`.
  - Trực quan hóa ma trận Attention trong Dịch máy. Hình: `attention.6007731a.png`.
- **Phần 3: Self-Attention \& Query-Key-Value (5 slides)**
  - Self-Attention: Từ tự chú ý đến chính nó trong cùng một câu để hiểu ngữ cảnh.
  - Trừu tượng hóa bằng Database: Cơ chế Truy vấn (Query) - Khóa (Key) - Giá trị (Value). Hình: `query-key-value.b57cceb0.png`.
  - Toán học của Attention: Bóc tách phương trình $Attention(Q, K, V)$.
  - Multi-Head Attention: Chú ý nhiều khía cạnh (Ngữ pháp, Cảm xúc, Đại từ). Hình: `multi-head-attention.718456ad.png`.
- **Phần 4: Kỷ nguyên Transformer (4 slides)**
  - Bài báo vĩ đại "Attention Is All You Need" (2017) - Khai tử RNN/LSTM.
  - Sơ đồ Encoder - Decoder của Transformer. Hình: `encoder-decoder.d979dbbc.png`.
  - Tại sao Transformer không hiểu thứ tự? Positional Encoding giải quyết vấn đề.
  - Tác động: Kỷ nguyên của LLM (Large Language Models) như GPT, BERT.
- **Tổng kết (1 slide)**

## 3. Các bước Triển khai
1. **Viết mã LaTeX:** Khởi tạo `TaiLieu/slideDL/Chapter15.tex`.
2. **Biên dịch:** Chạy `pdflatex` 2 lần liên tiếp.
3. **Hoàn thiện Tasks:** Báo cáo kết quả và cập nhật hệ thống.
