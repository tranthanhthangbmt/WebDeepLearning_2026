# Kế hoạch Xây dựng Slide Chương 18: LẬP TRÌNH XÁC SUẤT (Probabilistic Programming)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Sinh viên hiểu được sức mạnh của Lập trình Xác suất (Probabilistic Programming) - sự kết hợp giữa logic quan hệ, vũ trụ mở và các chương trình sinh tạo (Generative programs).
- **Yêu cầu kỹ thuật LaTeX:** 
  - Biên dịch chuẩn bằng `pdflatex` và `aima2e-slides.sty`.
  - Phân bổ hình ảnh minh họa cho các mô hình và mạng phức tạp (ví dụ: mô hình quan hệ).

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 18: LẬP TRÌNH XÁC SUẤT
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Liệt kê các đề mục từ 18.1 đến 18.4.

### 18.1 Các mô hình Xác suất Quan hệ (Relational Probability Models - RPM)
- Tại sao cần RPM? Khắc phục sự cứng nhắc của Mạng Bayes tĩnh khi xử lý các đối tượng có quan hệ với nhau.
- Cú pháp và ngữ nghĩa: Tích hợp logic Bậc nhất (First-Order Logic) vào Xác suất.
- *Ví dụ:* Đánh giá cấp độ kỹ năng người chơi (Skill Rating trong các game online như Elo/TrueSkill).
- Suy diễn trong các mô hình xác suất quan hệ.

### 18.2 Các mô hình xác suất của vũ trụ mở (Open-Universe Probability Models)
- Khái niệm Vũ trụ Mở (Open-Universe): Số lượng đối tượng không được biết trước (khác với Vũ trụ Đóng).
- Cú pháp và ngữ nghĩa: Định nghĩa xác suất trên sự tồn tại của các đối tượng (ví dụ số lượng khách hàng tiềm năng).
- Suy diễn trong mô hình Vũ trụ Mở (Gặp khó khăn do không gian tìm kiếm có thể tiến tới vô tận).
- Các ví dụ kinh điển.

### 18.3 Theo dõi trạng thái của một thế giới phức tạp
- Khi thế giới động và đối tượng liên tục xuất hiện/biến mất.
- **Ví dụ 1: Theo dõi đa mục tiêu (Multitarget tracking)** trong Radar hoặc Camera. (Bài toán Liên kết dữ liệu - Data Association).
- **Ví dụ 2: Theo dõi giao thông (Traffic monitoring)**.
- Thuật toán cốt lõi: Bộ lọc hạt mở rộng kết hợp đối sánh dữ liệu.

### 18.4 Chương trình đóng vai trò Mô hình Xác suất (Programs as Probability Models)
- Giới thiệu các ngôn ngữ Lập trình Xác suất (ví dụ: Church, Stan, Pyro).
- Một chương trình phần mềm (Program) không chỉ sinh ra một kết quả, mà nó \textit{sinh ra một phân phối xác suất}.
- *Ví dụ minh họa:* Xử lý quá trình đọc văn bản bị nhiễu (OCR).
- Cải thiện chương trình sinh tạo để kết hợp với mô hình Markov và cách suy luận xấp xỉ bên trong một ngôn ngữ lập trình.

### Tóm tắt Chương 18
- Lập trình xác suất cho phép lập trình viên định nghĩa các mô hình cực kỳ tinh vi chỉ bằng vài dòng code, ẩn giấu mọi phép toán suy luận phức tạp (MCMC, Gibbs Sampling) vào sâu bên trong trình biên dịch.

## 3. Kế hoạch Hiện thực hóa (Thực thi)

1. Việt hóa chính xác thuật ngữ "Open-universe", "Relational Probability Models".
2. Các ví dụ về "Theo dõi đa mục tiêu" và "Đánh giá cấp độ người chơi" sẽ được thiết kế ngắn gọn, đi thẳng vào ứng dụng thực tiễn để tăng hứng thú cho sinh viên.
3. Rà soát, biên dịch `pdflatex -interaction=nonstopmode Chapter18_4th.tex` để xác nhận thành phẩm.

---
**Ghi chú:** Kế hoạch Chương 18 hoàn tất. Tôi sẽ tiếp tục xuất bản kế hoạch cho Chương 19 ngay sau đây.
