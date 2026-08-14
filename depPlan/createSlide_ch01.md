# Kế hoạch Tạo Slide Chương 1: Khái niệm Học sâu (Deep Learning)

## 1. Phân tích Hiện trạng (Từ `TaiLieu/slide_4th`)
- **Cấu trúc cũ:** Sử dụng documentclass `article` kết hợp với một gói custom là `aima2e-slides.sty` (được mượn từ sách AI cũ) để định dạng trang thành dạng slide. Script `generate_tex.py` cũ có đường dẫn hardcode sang thư mục của môn Trí tuệ nhân tạo.
- **Vấn đề:** Gói `aima2e-slides.sty` không có sẵn trong dự án hiện tại (`TaiLieu/slide/` không tồn tại). Hơn nữa, định dạng `article` giả lập slide không còn phù hợp với tiêu chuẩn hiện đại, khó tùy biến và không tự động co giãn tốt nội dung chuyên nghiệp.

## 2. Đề xuất Thiết kế Mới cho Môn Học Sâu
Để đạt chuẩn **bài giảng bậc đại học**, chuyên nghiệp và hiện đại, tôi đề xuất sử dụng **LaTeX Beamer** (Class tiêu chuẩn toàn cầu cho việc tạo slide học thuật).
- **Theme:** Sử dụng theme hiện đại (ví dụ: `Metropolis` hoặc `Madrid` tùy chỉnh màu sắc).
- **Bố cục:** Tự động chia cột (chữ một bên, hình một bên) đối với các slide có hình ảnh để tận dụng tối đa không gian.
- **Hình ảnh:** Bắt buộc đưa toàn bộ 9 hình ảnh của chương 1 (từ thư mục `images/ch01/`) vào các slide tương ứng. Hình ảnh sẽ tự động căn giữa và thu phóng vừa vặn khung hình (keepaspectratio).

## 3. Cấu trúc Slide Chương 1 Dự kiến
- **Slide 1 (Tiêu đề):** Khái niệm Học sâu (What is deep learning?). Tên môn học, tên giảng viên.
- **Slide 2 (Mục lục):** Nội dung chính của chương 1.
- **Slide 3 - 5 (Khái niệm AI, ML, DL):** Giới thiệu trí tuệ nhân tạo, học máy và học sâu. Kèm hình ảnh `ai-ml-dl.07201556.png`.
- **Slide 6 (Mô hình lập trình mới):** Học máy khác biệt thế nào. Kèm hình ảnh `a-new-programming-paradigm.e8d1a1c2.png`.
- **Slide 7 - 9 (Học biểu diễn dữ liệu):** Không gian biểu diễn. Kèm hình `learning_representations.97fa3c4b.png` và `example_data_points.28a84f5a.png`.
- **Slide 10 - 13 (Cơ chế hoạt động của DL):** Khái niệm layer, trọng số (weights), hàm mất mát (loss function) và bộ tối ưu hóa (optimizer). Kèm chuỗi 3 hình `deep-learning-in-3-figures-*.png`.
- **Slide 14 - 15 (Thành tựu và Tương lai):** Các bước tiến lịch sử và tầm nhìn.

## 4. Các bước Triển khai Thực tế
1. **Khởi tạo:** Tạo thư mục `TaiLieu/slideDL/`.
2. **Soạn thảo:** Viết trực tiếp file `TaiLieu/slideDL/Chapter01.tex` sử dụng Beamer, tổng hợp nội dung văn bản tinh túy nhất từ bản dịch `chapters/chapter_01.md`.
3. **Biên dịch:** Sử dụng lệnh `pdflatex` (hoặc `latexmk`) để biên dịch file `.tex` sang `Chapter01.pdf`.
4. **Đánh giá:** Kiểm tra chất lượng hiển thị PDF, đảm bảo font tiếng Việt hiển thị hoàn hảo và toàn bộ 9 ảnh đều xuất hiện sắc nét.
