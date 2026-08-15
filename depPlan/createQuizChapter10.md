# Kế hoạch tạo bài tập trắc nghiệm (Interactive Quiz) cho Chương 10

## 1. Mục tiêu
- Xây dựng hệ thống câu hỏi trắc nghiệm tương tác cho **Chương 10: Diễn giải những gì ConvNet học được (Interpreting what ConvNets learn)**.
- Tích hợp giao diện UI tương tự như đã làm ở các chương trước vào bài học Markdown, giúp sinh viên củng cố kiến thức trực tiếp.

## 2. Chủ đề câu hỏi (30 câu)
Dựa trên nội dung của Chương 10, các câu hỏi sẽ xoay quanh các chủ đề:
1. **Khái niệm và Tầm quan trọng của Interpretability:** Tại sao cần diễn giải mô hình (ứng dụng y tế, hiểu quyết định phân loại thay vì "hộp đen").
2. **Trực quan hóa kích hoạt trung gian (Intermediate activations):**
   - Cách tạo Keras Model nhiều đầu ra (multi-output).
   - Đặc điểm của các lớp đầu (giữ nhiều thông tin hình ảnh nguyên bản, phát hiện cạnh).
   - Đặc điểm của các lớp sâu (trừu tượng hóa, mã hóa lớp, tăng tính thưa thớt - sparsity).
   - Sự tương đồng với nhận thức con người (lọc thông tin không liên quan).
3. **Trực quan hóa các bộ lọc ConvNet (Filter Visualization):**
   - Kỹ thuật Gradient Ascent trong không gian đầu vào (tối đa hóa đầu ra của bộ lọc).
   - So sánh TF, PyTorch, JAX trong việc tính gradient.
   - Trick chuẩn hóa gradient (L2 norm).
   - Ý nghĩa của pattern học được (tương tự Fourier transform).
4. **Grad-CAM (Class Activation Map):**
   - Tác dụng (xác định phần nào của ảnh đóng góp lớn nhất vào quyết định lớp).
   - Trọng số hóa bản đồ đặc trưng kênh bằng gradient của lớp dự đoán.
   - Xử lý và chồng bản đồ nhiệt (heatmap) lên ảnh gốc.

## 3. Cấu trúc và Giao diện (Thư mục `quizzes/Chapter10`)
- **`index.html`**: File giao diện chính, được nhúng qua iframe vào `Chapters/chapter_10.md`. Cập nhật tiêu đề thành Chương 10.
- **`style.css`**: Dùng chung phong cách (giữ nguyên).
- **`script.js`**: Logic xử lý UI (giữ nguyên).
- **`questions.js`**: Chứa 30 câu hỏi dưới định dạng mảng JSON. 
  - Tuân thủ nguyên tắc: **Chiều dài các đáp án phải tương đương nhau**. Giải thích chi tiết cho đáp án đúng.
  - Phân bổ: 25 Multiple Choice (MCQ), 2 Matching, 1 Sorting, 2 Fill-in-the-blank.
  - Phải có dòng `export default questions;` ở cuối.

## 4. Các bước thực hiện
1. Tạo thư mục `quizzes/Chapter10`.
2. Sao chép `index.html`, `style.css`, `script.js` từ `Chapter09` sang `Chapter10` và điều chỉnh nội dung text HTML.
3. Tạo file `questions.js` với 30 câu hỏi chi tiết về Chương 10.
4. Cập nhật `Chapters/chapter_10.md` bằng cách chèn thẻ `iframe` của quiz vào cuối phần "Tiếng Việt", ngay trước thẻ `<!-- tabs:end -->`.
5. Kiểm tra tính toàn vẹn (lỗi cú pháp, logic).
