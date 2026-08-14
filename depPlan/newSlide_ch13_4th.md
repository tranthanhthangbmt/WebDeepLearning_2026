# Kế hoạch Xây dựng Slide Chương 13: SUY DIỄN XÁC SUẤT (Probabilistic Reasoning)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Tổng hợp và trình bày các kiến thức cốt lõi về Suy diễn xác suất, Mạng Bayes (Bayesian Networks) và Mạng nhân quả từ sách AIMA 4th edition.
- **Yêu cầu kỹ thuật LaTeX:** 
  - Đảm bảo biên dịch hoàn hảo trên `pdflatex`.
  - Cấu trúc Mạng Bayes cần biểu diễn bằng hình ảnh đồ thị rõ nét kết hợp các công thức toán học tính xác suất có điều kiện. Các ký hiệu toán như $\mathbf{P}$, $\sum$, $\prod$, và các chỉ số dưới/trên phải căn lề và hiển thị chuẩn học thuật đại học.

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 13: SUY DIỄN XÁC SUẤT
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Liệt kê các danh mục cốt lõi từ 13.1 đến 13.5.

### 13.1 Biểu diễn Kiến thức trong một Miền Không chắc chắn
- Giới thiệu nhu cầu nén không gian biến ngẫu nhiên.
- Mạng Bayes là gì? (Đồ thị có hướng không chu trình - DAG) kết hợp với các Bảng xác suất có điều kiện (CPT).
- *Ví dụ kinh điển:* Bài toán Báo trộm (Burglary, Earthquake, Alarm, JohnCalls, MaryCalls).
- *Hình ảnh dự kiến:* `figure_13.1.jpg` (Đồ thị mạng Bayes) và `figure_13.2.jpg` (Bảng CPT tương ứng).

### 13.2 Ngữ nghĩa của Mạng Bayes
- **Ngữ nghĩa Toàn cục (Global Semantics):** Biểu diễn phân phối đồng thời đầy đủ (Full Joint Distribution) thông qua tích các phân phối cục bộ: $\mathbf{P}(X_1, ..., X_n) = \prod_{i=1}^{n} \mathbf{P}(X_i | \text{Parents}(X_i))$.
- **Các quan hệ độc lập có điều kiện:** Khái niệm d-separation.
- Biểu diễn hiệu quả của CPT (Ví dụ dùng Noisy-OR để giảm số lượng tham số từ $O(2^k)$ xuống $O(k)$).
- Các mạng Bayes với các biến liên tục và Nghiên cứu tình huống về Bảo hiểm Ô tô (Car insurance).

### 13.3 Suy luận Chính xác trong các Mạng Bayes
- Bài toán suy luận: Tính $\mathbf{P}(X | \mathbf{e})$.
- **Suy luận bằng phép liệt kê (Inference by Enumeration):** Triển khai dạng cây, dễ cài đặt nhưng độ phức tạp $O(2^n)$.
- **Thuật toán Loại bỏ Biến (Variable Elimination):** Tối ưu hóa tính toán bằng cách đẩy dấu tổng $\sum$ vào sâu bên trong và dùng kỹ thuật lưu trữ động (Factors).
- Độ phức tạp của suy luận chính xác: NP-hard đối với mạng đa liên kết.
- Các thuật toán phân cụm (Clustering algorithms / Junction Tree algorithm).

### 13.4 Suy luận Xấp xỉ cho các Mạng Bayes
- Tại sao cần suy luận xấp xỉ? (Vì Variable Elimination quá chậm cho mạng Bayes khổng lồ).
- **Các phương pháp lấy mẫu trực tiếp (Direct Sampling):** Rejection Sampling (Lấy mẫu loại bỏ) và Likelihood Weighting (Lấy mẫu theo trọng số Likelihood).
- **Suy luận bằng mô phỏng Chuỗi Markov (MCMC):** Thuật toán Gibbs Sampling để lặp qua lặp lại các trạng thái và hội tụ về xác suất thực tế.
- *Hình ảnh dự kiến:* Biểu đồ hội tụ lỗi của MCMC so với Likelihood Weighting.

### 13.5 Các Mạng Nhân quả (Causal Networks)
- Sự khác biệt giữa Mối tương quan (Correlation) và Nhân quả (Causation).
- Biểu diễn các hành động can thiệp bằng **Toán tử *do*** ($do(X=x)$).
- **Tiêu chuẩn cửa sau (The back-door criterion):** Một trong những công cụ toán học mạnh mẽ nhất để triệt tiêu biến nhiễu (confounders) khi đánh giá nhân quả.

### Tóm tắt Chương 13
- Tầm quan trọng của Mạng Bayes trong việc biến một phân phối khổng lồ $O(2^n)$ thành đồ thị trực quan và khả thi để tính toán.
- Sự kết hợp hoàn hảo giữa cấu trúc đồ thị nhân quả và số liệu xác suất thống kê.

## 3. Kế hoạch Hiện thực hóa (Thực thi)

1. Rà soát ngôn từ chuẩn học thuật (d-separation, Likelihood Weighting, Variable Elimination).
2. Xây dựng slide bằng cách bóc tách từng bullet point, dùng \blob để liệt kê và dùng môi trường toán học \begin{itemize} ... $ ... $ để lồng ghép công thức Bayes.
3. Chèn khối hình ảnh Đồ thị mạng Bayes một cách logic, ưu tiên hiển thị toàn màn hình các cây liệt kê thuật toán Variable Elimination để sinh viên tiện theo dõi từng bước tính toán (factors).
4. Thực thi kiểm định cuối cùng: Biên dịch bằng lệnh `pdflatex -interaction=nonstopmode Chapter13_4th.tex` để xác nhận file PDF hoàn hảo không lỗi font.

---
**Ghi chú:** Đây là kế hoạch chuyên môn cho Chương 13. Thầy hãy kiểm tra độ chính xác và tính hàn lâm. Nếu thầy "Duyệt", tôi sẽ tiếp tục trích xuất và lên kế hoạch luôn cho **Chương 14**.
