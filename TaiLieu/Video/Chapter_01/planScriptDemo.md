# Kế Hoạch Tạo Script Demo - Chương 1 (Trí Tuệ Nhân Tạo: Giới Thiệu)

## 1. Mục Tiêu
Tạo một kịch bản (script) chi tiết phục vụ cho việc tạo video/audio bài giảng Chương 1 - Môn Máy học. Script sẽ mô phỏng một buổi học trên lớp bậc đại học, dưới dạng cuộc hội thoại trao đổi hai chiều giữa Giảng viên (Người 1) và Sinh viên (Người 2), nhằm giúp sinh viên hiểu sâu lý thuyết, phương pháp và ví dụ thực tiễn.

## 2. Dữ Liệu Đầu Vào
- **File Slide PDF:** `slideML/Slide_ML_Chap01.pdf` (Chứa các slide trình chiếu của Chương 1).
- **Nội Dung Chi Tiết:** `machineLearningWeb/docs/chuong_01.md` (Chứa phần lý thuyết nền tảng, công thức và giải thích chi tiết cho từng phần của chương).

## 3. Cấu Trúc Thư Mục & File Đầu Ra
- **Kiểm tra và tạo thư mục:** Nếu thư mục `machineLearningWeb/Video/Chapter_01` chưa tồn tại, hệ thống (hoặc người thực hiện) sẽ tạo mới thư mục này.
- **Copy File:** Copy file `Slide_ML_Chap01.pdf` từ thư mục `slideML` vào thư mục `machineLearningWeb/Video/Chapter_01`.
- **File Script:** Tạo và lưu file script (ví dụ: `scriptCh01.txt` hoặc cập nhật `scriptDemo.txt`) trong thư mục `machineLearningWeb/Video/Chapter_01`.

## 4. Nguyên Tắc Xây Dựng Kịch Bản (Script)
1. **Định dạng nhân vật:** 
   - **Người 1 (Thầy/Cô):** Đóng vai trò dẫn dắt, đưa ra lý thuyết, phương pháp, đặt câu hỏi gợi mở và tổng kết vấn đề.
   - **Người 2 (Sinh viên):** Đóng vai trò tiếp thu, đặt câu hỏi thắc mắc, trả lời gợi mở của giảng viên, và tóm tắt lại ý hiểu của mình.
2. **Tiến trình bài giảng:** 
   - Kịch bản đi theo từng slide một (Slide 1, Slide 2,... cho đến hết).
   - Nội dung thảo luận trên mỗi slide được trích xuất và tham chiếu sâu từ file `chuong_01.md` để đảm bảo độ chi tiết như một tiết học thực tế ở bậc đại học.
3. **Mức độ chi tiết:** 
   - Không chỉ đọc lại slide. Cần diễn giải lý thuyết, phân tích các phương pháp, và đặc biệt là lồng ghép các **ví dụ minh họa** thực tế giúp sinh viên dễ hiểu.
4. **Xử lý công thức toán học:**
   - **Bắt buộc:** Chuyển đổi toàn bộ công thức toán học, ký hiệu toán học sang **dạng văn bản (chữ đọc được)** vì script sẽ được dùng để tạo audio (Text-to-Speech).
   - *Ví dụ:* $x^2$ chuyển thành "x bình phương", $\sum_{i=1}^n$ chuyển thành "tổng từ i bằng 1 đến n", $f(x)$ chuyển thành "hàm f của x".

## 5. Các Bước Thực Hiện Cụ Thể
- **Bước 1 (Chuẩn bị không gian làm việc):** Khởi tạo thư mục `Chapter_01` trong `Video`, copy slide PDF sang thư mục mới.
- **Bước 2 (Phân tích dữ liệu):** Đọc nội dung Markdown `chuong_01.md` và đối chiếu với cấu trúc `Slide_ML_Chap01.pdf`. Mapping nội dung từng phần trong docs vào từng slide tương ứng.
- **Bước 3 (Soạn thảo kịch bản - Mở bài):** Giới thiệu môn học, tổng quan chương 1, tạo không khí lớp học.
- **Bước 4 (Soạn thảo kịch bản - Thân bài):** Triển khai viết script cho từng slide. Tại mỗi slide:
  - Thầy giáo (Người 1) mở đầu ý của slide.
  - Sinh viên (Người 2) phản hồi hoặc hỏi thêm.
  - Thầy giáo giải thích chi tiết, đọc rõ các công thức bằng chữ, đưa ra ví dụ (dựa trên `chuong_01.md`).
- **Bước 5 (Soạn thảo kịch bản - Kết luận):** Tổng kết chương, dặn dò, câu hỏi thảo luận cuối giờ.
- **Bước 6 (Review & Tinh chỉnh):** Đọc lại toàn bộ kịch bản để đảm bảo văn phong tự nhiên, tính liền mạch và các công thức toán đã hoàn toàn ở dạng văn bản đọc được.

## 6. Tiêu Chí Nghiệm Thu
- Thư mục `machineLearningWeb/Video/Chapter_01` đã có đủ slide và file script.
- Script tuân thủ đúng định dạng 2 nhân vật (Người 1, Người 2).
- Nội dung chi tiết, sâu sắc (có ví dụ, lý thuyết từ `docs`).
- Không còn sót ký hiệu toán học dạng raw (như LaTeX hay symbol) trong phần lời thoại.
