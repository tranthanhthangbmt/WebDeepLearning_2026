# Kế hoạch Xây dựng Slide Chương 11: LẬP KẾ HOẠCH TỰ ĐỘNG (Automated Planning)

## 1. Mục tiêu và Tiêu chuẩn
- **Mục tiêu:** Chuyển hóa nội dung từ `chapter_11_vi.html` thành slide LaTeX mang tính học thuật cao, phục vụ giảng dạy đại học (thời lượng ~90 phút).
- **Yêu cầu kỹ thuật:** 
  - Đảm bảo biên dịch thành công với `pdflatex`.
  - Tái sử dụng trọn vẹn macro `\heading` đã được chuẩn hóa để text không bị tràn viền.
  - Sử dụng hệ thống toán học tiêu chuẩn (không dùng `amsmath` để tránh xung đột `\implies` của package `aima2e-slides.sty`).
  - Gắn hình ảnh sát với lý thuyết: Định dạng ảnh `width=0.95\textwidth, height=0.65\textheight, keepaspectratio` hoặc tùy chỉnh phù hợp để ảnh hiển thị chuyên nghiệp.

## 2. Cấu trúc Nội dung dự kiến (Mapping Sections)

Toàn bộ chương 11 sẽ được chia thành các Heading rõ ràng, mỗi Heading đi kèm với mô tả nội dung đạn đạo (bullet points) và hình ảnh:

### Trang Tiêu đề
- **Tiêu đề chính:** CHƯƠNG 11: LẬP KẾ HOẠCH TỰ ĐỘNG
- **Tiêu đề phụ:** Trí tuệ nhân tạo - Artificial Intelligence
- **Nội dung Chương:** Liệt kê các mục chính từ 11.1 đến 11.7.

### 11.1 Định nghĩa về Lập kế hoạch Cổ điển (Classical Planning)
- Giới thiệu ngôn ngữ PDDL (Planning Domain Definition Language).
- Định nghĩa Trạng thái (States), Hành động (Actions), Tiền điều kiện (Preconditions) và Hiệu ứng (Effects).
- **Ví dụ kinh điển:** Bài toán Vận chuyển hàng không (Air cargo), Lốp dự phòng (Spare tire) và Thế giới khối vuông (Blocks world).
- *Hình ảnh dự kiến:* `figure_11.1.jpg` đến `figure_11.4.jpg` (Minh họa ngôn ngữ PDDL hoặc Blocks world).

### 11.2 Các thuật toán Lập kế hoạch Cổ điển
- Tìm kiếm không gian trạng thái Tiến (Forward state-space search).
- Tìm kiếm Lùi (Backward search) phân tích sự phù hợp của các Preconditions.
- Biểu diễn việc lập kế hoạch dưới dạng Bài toán Thỏa mãn Boolean (SAT) hoặc Mạng kế hoạch (Planning Graph).
- *Hình ảnh dự kiến:* `figure_11.5.jpg`, `figure_11.6.jpg` (Sơ đồ duyệt không gian trạng thái).

### 11.3 Các hàm tự suy (Heuristics) cho Lập kế hoạch
- Làm thế nào AI tính được hàm $h(s)$ để hướng dẫn quá trình lập kế hoạch?
- **Khái niệm Relaxation:** Bỏ qua bớt các Preconditions hoặc Negative effects để bài toán dễ giải hơn.
- Cắt tỉa nhánh độc lập với miền (Domain-independent pruning).
- *Hình ảnh dự kiến:* Sơ đồ minh họa Heuristics `figure_11.7.jpg` (nếu có).

### 11.4 Lập kế hoạch Phân cấp (Hierarchical Planning)
- Chia nhỏ một nhiệm vụ phức tạp (Ví dụ: Xây nhà) thành các hành động cấp cao (High-level actions - HLA).
- Quá trình tinh chỉnh (Refinement) để biến HLA thành các Hành động nguyên thủy (Primitive actions).
- *Hình ảnh dự kiến:* `figure_11.8.jpg`, `figure_11.9.jpg` (Cây phân cấp các hành động).

### 11.5 Lập kế hoạch trong Môi trường Không tất định
- Khi thế giới không chắc chắn hoặc Cảm biến không hoàn hảo (Sensorless planning / Conformant planning).
- Lập kế hoạch dự phòng (Contingent planning) - Xây dựng kế hoạch theo cấu trúc rẽ nhánh (Nếu/Thì).
- Lập kế hoạch trực tuyến (Online planning) và Giám sát thực thi.
- *Hình ảnh dự kiến:* `figure_11.10.jpg`, `figure_11.11.jpg` (Đồ thị các trạng thái niềm tin - Belief states).

### 11.6 Thời gian, Lịch trình và Tài nguyên
- Biểu diễn các ràng buộc song song và thứ tự thời gian.
- **Phương pháp Đường găng (Critical Path Method - CPM)**: Tính ES (Earliest Start), LS (Latest Start) và Slack.
- **Ràng buộc tài nguyên:** Lập lịch biểu (Job-shop scheduling) với nguồn tài nguyên hữu hạn.
- *Hình ảnh dự kiến:* `figure_11.14.jpg`, `figure_11.15.jpg` (Biểu đồ tiến độ Gantt chart / Đường găng).

### Tóm tắt Chương 11
- Rút ra những kết luận cốt lõi về bản chất của hệ thống Lập kế hoạch (Planning).
- Sự tiến hóa từ Tìm kiếm thuần túy $\rightarrow$ Logic Bậc Nhất $\rightarrow$ Ngôn ngữ Lập kế hoạch PDDL $\rightarrow$ Lịch trình thời gian thực.

## 3. Kế hoạch Hiện thực hóa (Thực thi)

1. Tự động parse và tổng hợp nội dung chi tiết từ tư duy sách giáo trình AIMA, viết trực tiếp bằng tiếng Việt chuẩn đại học.
2. Dùng các câu lệnh LaTeX chuẩn: `\blob` để tạo gạch đầu dòng, `\textbf{...}` để nhấn mạnh các từ khóa hàn lâm.
3. Chèn khối hình ảnh (`\begin{figure}`) một cách có tính toán, mỗi hình ảnh chiếm xấp xỉ một trang nếu cần để đảm bảo sinh viên ngồi cuối giảng đường vẫn thấy rõ.
4. Kiểm duyệt lại file `.tex`, chạy thử lệnh biên dịch `pdflatex -interaction=nonstopmode Chapter11_4th.tex` để xác nhận thành công trước khi hoàn thành công việc.

---
**Ghi chú:** Đây là kế hoạch chuyên sâu cho Chapter 11. Sau khi thầy phản hồi xác nhận, tôi sẽ bắt tay vào biên dịch code LaTeX cho Chapter 11 ngay lập tức, rồi sau đó sẽ lập kế hoạch tương tự cho Chương 12.
