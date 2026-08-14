# Kế hoạch Xây dựng Slide Chương 21: HỌC CÁC MÔ HÌNH XÁC SUẤT (Learning Probabilistic Models)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Giới thiệu nền tảng thống kê trong học máy. Trọng tâm là Phương pháp Ước lượng hợp lý cực đại (MLE), Ước lượng Bayes, và siêu thuật toán EM (Expectation-Maximization) dành cho biến ẩn.
- **Yêu cầu kỹ thuật LaTeX:** 
  - Biên dịch 100% sạch sẽ bằng `pdflatex`.
  - Thiết lập các phương trình Log-Likelihood ($L(\theta)$, $\arg\max$), tính toán đạo hàm một phần ($\frac{\partial L}{\partial \theta}$) theo format hàn lâm, tránh tràn lề.

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 21: HỌC CÁC MÔ HÌNH XÁC SUẤT
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Liệt kê các danh mục cốt lõi từ 21.1 đến 21.3.

### 21.1 Học máy Thống kê (Statistical Learning)
- Từ bỏ khái niệm "Đúng/Sai" tuyệt đối để đến với mô hình Xác suất.
- Sự kiện quan sát được và bài toán tìm cấu trúc tham số (Parameter Learning) cho hệ thống.

### 21.2 Học với Dữ liệu Đầy đủ (Learning with Complete Data)
- **Học tham số hợp lý cực đại (Maximum Likelihood Estimation - MLE):**
  - Khái niệm: Tìm tham số $\theta$ để xác suất sinh ra tập dữ liệu quan sát là cao nhất.
  - Sử dụng Log-Likelihood để chuyển tích thành tổng, làm cho việc đạo hàm trở nên dễ dàng.
- Mô hình Naive Bayes: Biểu diễn MLE cho bài toán phân loại văn bản.
- Mô hình sinh (Generative models) vs. Mô hình phân biệt (Discriminative models).
- **Học tham số Bayes (Bayesian parameter learning):**
  - Khắc phục lỗi của MLE khi dữ liệu nhỏ (Ví dụ: Tung đồng xu 3 lần ra ngửa, MLE kết luận xác suất ngửa là 100%).
  - Áp dụng Định lý Bayes: $\mathbf{P}(\theta | data) \propto \mathbf{P}(data | \theta) \mathbf{P}(\theta)$.
  - Phân phối tiên nghiệm (Prior distribution) và phân phối Beta.
- Ước lượng mật độ với mô hình phi tham số (Nonparametric density estimation).

### 21.3 Học với các Biến ẩn: Thuật toán EM (Learning with Hidden Variables: The EM Algorithm)
- Tại sao cần EM? (Khi thế giới thực luôn thiếu sót dữ liệu - Missing Data, hoặc có những biến không thể quan sát - Hidden variables).
- **Thuật toán EM (Expectation-Maximization):**
  1. *Bước E (Expectation):* Giả định tham số hiện tại, tính toán giá trị kỳ vọng (xác suất) cho các biến ẩn.
  2. *Bước M (Maximization):* Dùng giá trị kỳ vọng vừa tính được như dữ liệu quan sát thật, chạy MLE để cập nhật lại tham số.
- **Ứng dụng 1: Hỗn hợp Gaussian (Mixtures of Gaussians):**
  - Giải bài toán gom cụm không giám sát (Clustering) dạng mềm (Soft-clustering) thay vì k-Means (Hard-clustering).
- **Ứng dụng 2: Học mạng Bayes với biến ẩn:** (Ví dụ: Học tham số cho Túi kẹo bị trộn lẫn - `figure_21.14`).
- **Ứng dụng 3: Học các mô hình Markov ẩn (HMM):** Thuật toán Baum-Welch.

### Tóm tắt Chương 21
- MLE là kỹ thuật vĩ đại của thống kê học. Nhưng Thuật toán EM mới thực sự mang lại sức mạnh cho AI trong các bài toán nhận dạng giọng nói, tin sinh học và khai phá dữ liệu không đầy đủ.

## 3. Kế hoạch Hiện thực hóa (Thực thi)

1. Việt hóa từ khóa: "Log-likelihood" $\rightarrow$ Khả năng Log, "Generative" $\rightarrow$ Mô hình sinh, "Expectation-Maximization" $\rightarrow$ Cực đại hóa Kỳ vọng (EM).
2. Trình bày Thuật toán EM một cách từ tốn, step-by-step. Mật độ chữ trên mỗi slide cho phần này phải đặc biệt thấp để nhường chỗ cho khối công thức toán học.
3. Chèn hình ảnh trực quan mô phỏng sự dịch chuyển của các đường cong Gaussian trong quá trình thuật toán EM tối ưu hóa (Mixtures of Gaussians).
4. Kiểm định lỗi `pdflatex -interaction=nonstopmode`.

---
**Ghi chú:** Đây là bản phác thảo chi tiết Chương 21. Kế hoạch Chương 22 cũng sẽ được triển khai ngay sau đây!
