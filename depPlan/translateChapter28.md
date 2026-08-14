# Kế hoạch chi tiết dịch Chương 28: PHILOSOPHY, ETHICS, AND SAFETY OF AI (Triết học, đạo đức và an toàn của AI)

## I. Mục tiêu
- Dịch nội dung **Chương 28: Philosophy, ethics, and safety of AI** từ sách tiếng Anh sang tiếng Việt.
- Bảo toàn cấu trúc HTML gốc, công thức toán học với MathJax (nếu có), thẻ danh sách, in đậm/nghiêng.
- Tự động tách nhỏ nội dung thành nhiều file để dễ kiểm soát tiến độ, sau đó gộp lại.
- Dò tìm và chèn đúng vị trí hình ảnh (Figure 28.1, 28.2,...) vào bản dịch.
- Tích hợp kết quả vào repository hiện tại để sinh viên dễ dàng tra cứu qua web.

## II. Phân chia công việc (Divide and Conquer)
Tổng cộng chương 28 có 31 trang, được chia thành **8 phần (từ part 1 đến part 8)** để đảm bảo tính ổn định và chi tiết trong quá trình dịch thuật. Các phần này tương ứng với thư mục lưu trữ `TaiLieu/ebooks_Chapters_Vi3/Chapter_28_Philosophy, ethics, and safety of AI/`.

- **Phần 1 (`part1_28.html`):** 28.1 The Limits of AI (28.1.1 The argument from informality, 28.1.2 The argument from disability, 28.1.3 The mathematical objection, 28.1.4 Measuring AI) và mở đầu 28.2 Can Machines Really Think?
- **Phần 2 (`part2_28.html`):** Tiếp tục 28.2 (28.2.1 The Chinese room, 28.2.2 Consciousness and qualia), 28.3 The Ethics of AI, 28.3.1 Lethal autonomous weapons.
- **Phần 3 (`part3_28.html`):** Tiếp tục 28.3 (28.3.2 Surveillance, security, and privacy, 28.3.3 Fairness and bias).
- **Phần 4 (`part4_28.html`):** Tiếp tục 28.3 (28.3.4 Trust and transparency).
- **Phần 5 (`part5_28.html`):** Tiếp tục 28.3 (28.3.5 The future of work, 28.3.6 Robot rights).
- **Phần 6 (`part6_28.html`):** Tiếp tục 28.3 (28.3.7 AI Safety).
- **Phần 7 (`part7_28.html`):** Tóm tắt (Summary) và Ghi chú Lịch sử - Thư mục (Bibliographical and Historical Notes).
- **Phần 8 (`part8_28.html`):** Phần còn lại của Ghi chú Lịch sử - Thư mục (Bibliographical and Historical Notes).

## III. Các bước thực hiện chi tiết

**Bước 1: Trích xuất và dịch từng phần (Translation phase)**
- Dùng kịch bản Python (`extract_pages_28.py`) để trích xuất văn bản từ tệp PDF thành các tệp `.txt` từ `part1_28.txt` đến `part8_28.txt` (đã thực hiện xong).
- Tiến hành dịch tuần tự: đọc từng file `partX_28.txt`, giữ nguyên ý nghĩa chuyên ngành và cấu trúc, xuất ra tệp `partX_28.html`.
- Áp dụng chặt chẽ định dạng thẻ HTML và MathJax cho công thức.
- Bắt giữ các chú thích hình ảnh định dạng `<p><strong>Hình 28.X</strong> ...</p>`.

**Bước 2: Gộp file (Merging)**
- Tạo kịch bản Python `merge_28.py` để ghép tất cả 8 file `partX_28.html` theo thứ tự từ 1 đến 8.
- Lưu file tổng hợp vào `TaiLieu/ebooks_Chapters_Vi3/Chapter_28_Philosophy, ethics, and safety of AI/chapter_28_vi.html`.

**Bước 3: Chèn hình ảnh (Image Injection)**
- Chạy kịch bản chèn hình ảnh (`inject_figures_all_3.py`) để dò tìm các thẻ `<strong>Hình 28.X</strong>` trong file `chapter_28_vi.html`.
- Thay thế và chèn mã HTML hiển thị ảnh lấy từ thư mục `TaiLieu/Figures/Images/figure_28.X.jpg` (nếu có).

**Bước 4: Tích hợp website (Integration)**
- Cập nhật tệp markdown `chapters/chapter_28_philosophy_ethics_and_safety_of_ai.md` (nếu tên tệp là như vậy) hoặc tệp tương ứng.
- Trỏ thẻ `<iframe>` trong mục `#### **Tiếng Việt**` đến đường dẫn chứa tệp `chapter_28_vi.html` (thêm tham số cache như `?v=1` nếu cần thiết).

**Bước 5: Kiểm tra (Verification)**
- Theo dõi bằng danh sách (Task List) và kiểm tra giao diện hiển thị web.
- Đảm bảo hình ảnh được căn giữa, văn bản dễ đọc và không có đoạn mã thừa.

---
**Một số lưu ý thuật ngữ:**
- Weak AI -> AI yếu
- Strong AI -> AI mạnh
- Embodied cognition -> Nhận thức hiện thân
- The Chinese room -> Căn phòng tiếng Trung
- Qualia -> Cảm giác chủ quan (Qualia)
- Lethal autonomous weapons -> Vũ khí sát thương tự động
- Fairness and bias -> Sự công bằng và thiên vị
- Trust and transparency -> Niềm tin và sự minh bạch
- Value alignment problem -> Vấn đề căn chỉnh giá trị
