# Kế Hoạch Tạo Script - Chương 2: Toán học của NN

## 1. Mục Tiêu
Tạo một kịch bản (script) chi tiết phục vụ cho việc tạo video/audio bài giảng Chương 2 - Môn Học sâu. Script sẽ mô phỏng một buổi học trên lớp bậc đại học, dưới dạng cuộc hội thoại trao đổi hai chiều giữa Giảng viên (Người 1) và Sinh viên (Người 2), nhằm giúp sinh viên hiểu sâu lý thuyết, phương pháp và ví dụ thực tiễn. Số lượng slide trong script phải **bằng đúng số lượng slide** trong file slide PDF (tương đương 34 slide).

## 2. Dữ Đầu Vào
- **File Slide PDF/Tex:** `TaiLieu/slideDL/Chapter02.pdf` (hoặc cấu trúc trong `Chapter02.tex` gồm 34 frame).
- **Nội Dung Chi Tiết:** Tab **Tiếng Việt** của file `Chapters/chapter_02.md` (Chứa phần lý thuyết nền tảng, công thức và giải thích chi tiết cho từng phần của chương).

## 3. Cấu Trúc Thư Mục & File Đầu Ra
- **Thư mục làm việc:** `TaiLieu/Video/Chapter_02`
- **Copy File:** Copy file `Chapter02.pdf` từ thư mục `TaiLieu/slideDL/` vào thư mục `TaiLieu/Video/Chapter_02`.
- **File Script:** Tạo và lưu file script với tên `scriptCh02.txt` trong thư mục `TaiLieu/Video/Chapter_02`.

## 4. Nguyên Tắc Xây Dựng Kịch Bản (Script)
1. **Định dạng nhân vật:** 
   - **Người 1 (Thầy giáo/Giảng viên):** Đóng vai trò dẫn dắt, đưa ra lý thuyết, phương pháp, đặt câu hỏi gợi mở và tổng kết vấn đề.
   - **Người 2 (Sinh viên):** Đóng vai trò tiếp thu, học bài, đặt câu hỏi thắc mắc, trả lời gợi mở của giảng viên, và tóm tắt lại ý hiểu của mình.
2. **Tiến trình bài giảng:** 
   - Kịch bản đi theo từng slide một (từ Slide 1 đến Slide 34). **Không được bỏ sót hay gộp bất kỳ slide nào**.
   - Nội dung thảo luận trên mỗi slide được trích xuất và tham chiếu sâu từ nội dung **tab Tiếng Việt** của file `chapter_02.md` để đảm bảo độ chi tiết như một tiết học thực tế ở bậc đại học. Cần bám sát cả ý chính trên Slide và giải nghĩa trong tài liệu Markdown.
3. **Mức độ chi tiết:** 
   - Không chỉ đọc lại các gạch đầu dòng trên slide. Cần diễn giải lý thuyết, phân tích các phương pháp, và đặc biệt là lồng ghép các **ví dụ minh họa** thực tế giúp sinh viên dễ hiểu.
4. **Xử lý công thức toán học:**
   - **Bắt buộc:** Chuyển đổi toàn bộ công thức toán học, ký hiệu toán học sang **dạng văn bản (viết thành lời đọc)** vì script sẽ được dùng để tạo audio (Text-to-Speech).
   - *Ví dụ:* $x^2$ chuyển thành "x bình phương", $\sum_{i=1}^n$ chuyển thành "tổng từ i bằng 1 đến n", $f(x)$ chuyển thành "hàm f của x", $X > 0$ chuyển thành "ích lớn hơn không".

## 5. Các Bước Thực Hiện Cụ Thể
- **Bước 1 (Phân tích dữ liệu):** Đọc nội dung tab Tiếng Việt của `chapter_02.md` và đối chiếu với cấu trúc 34 slide trong `Chapter02.tex`. Map nội dung từng phần trong tài liệu vào từng slide tương ứng.
- **Bước 2 (Soạn thảo kịch bản - Mở bài - Slide 1 & 2):** Giới thiệu chương học mới, tổng quan chương, tạo không khí lớp học.
- **Bước 3 (Soạn thảo kịch bản - Thân bài - Slide 3 đến 33):** Triển khai viết script cho từng slide. Tại mỗi slide:
  - Thầy giáo (Người 1) mở đầu ý của slide.
  - Sinh viên (Người 2) phản hồi hoặc hỏi thêm để đào sâu vấn đề.
  - Thầy giáo giải thích chi tiết, đọc rõ các công thức bằng chữ, đưa ra ví dụ thực tiễn (dựa trên tài liệu tiếng Việt của `chapter_02.md`).
- **Bước 4 (Soạn thảo kịch bản - Kết luận - Slide 34):** Tổng kết chương, dặn dò sinh viên chuẩn bị cho bài sau.
- **Bước 5 (Review & Tinh chỉnh):** Đọc lại toàn bộ kịch bản để đảm bảo văn phong tự nhiên, tính liền mạch, kiểm tra đã đủ 34 slide và xác nhận các công thức toán đã hoàn toàn ở dạng văn bản đọc thành lời.

## 6. Tiêu Chí Nghiệm Thu
- File `scriptCh02.txt` được tạo thành công trong thư mục `TaiLieu/Video/Chapter_02`.
- Script có **chính xác 34 slide**, có đánh dấu rõ ràng từng Slide (ví dụ: `Slide 1: ...`).
- Tuân thủ đúng định dạng 2 nhân vật (`Người 1:`, `Người 2:`).
- Nội dung bám sát lý thuyết cấp bậc đại học, có ví dụ minh họa lấy từ tab Tiếng Việt của `chapter_02.md`.
- Tuyệt đối không còn ký hiệu toán học dạng raw (như LaTeX `$`, `\sum`) trong phần lời thoại. Mọi công thức đã được phiên âm thành chữ.
