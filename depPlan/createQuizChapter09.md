# Kế hoạch tạo bài tập trắc nghiệm (Interactive Quiz) cho Chương 9

## 1. Mục tiêu
- Xây dựng hệ thống câu hỏi trắc nghiệm tương tác cho **Chương 9: Các mẫu kiến trúc ConvNet**.
- Tích hợp giao diện UI tương tự như đã làm ở các chương trước vào bài học Markdown, giúp sinh viên củng cố kiến thức trực tiếp.

## 2. Chủ đề câu hỏi (30 câu)
Dựa trên nội dung của Chương 9, các câu hỏi sẽ xoay quanh các chủ đề:
1. **Tính mô đun, phân cấp và tái sử dụng (MHR):** Khái niệm, cách thức tổ chức cấu trúc của deep learning, sự phát triển phân cấp đặc trưng.
2. **Kết nối dư (Residual connections):** Vấn đề biến mất đạo hàm (vanishing gradients), cấu trúc residual block, shortcut, chiếu tuyến tính (linear projection) với Conv 1x1.
3. **Chuẩn hóa hàng loạt (Batch Normalization):** Khái niệm, tác dụng (giảm covariant shift, lan truyền gradient), vị trí đặt BatchNormalization (trước/sau Activation), thuộc tính use_bias.
4. **Tích chập có thể phân tách theo chiều sâu (Depthwise Separable Convolutions):** Cấu tạo (depthwise + pointwise 1x1), ưu điểm (giảm tham số, nhẹ hơn), giả định tính độc lập giữa các kênh (channels).
5. **Kiến trúc Xception & Mini Xception:** Sự kết hợp các kiến thức trên vào một mô hình thực tế.
6. **Vision Transformers (ViTs):** Sự khác biệt giữa ViT và ConvNet, xử lý tuần tự patch hình ảnh, lượng dữ liệu yêu cầu.

## 3. Cấu trúc và Giao diện (Thư mục `quizzes/Chapter09`)
- **`index.html`**: File giao diện chính, được nhúng qua iframe vào `Chapters/chapter_09.md`. Cập nhật tiêu đề thành Chương 9.
- **`style.css`**: Dùng chung phong cách (giữ nguyên).
- **`script.js`**: Logic xử lý UI (giữ nguyên).
- **`questions.js`**: Chứa 30 câu hỏi dưới định dạng mảng JSON. 
  - Tuân thủ nguyên tắc: **Chiều dài các đáp án phải tương đương nhau**. Giải thích cực kỳ chi tiết cho đáp án đúng.
  - Phân bổ: 25 Multiple Choice (MCQ), 2 Matching, 1 Sorting, 2 Fill-in-the-blank.
  - Phải có dòng `export default questions;` ở cuối.

## 4. Các bước thực hiện
1. Tạo thư mục `quizzes/Chapter09`.
2. Sao chép `index.html`, `style.css`, `script.js` từ `Chapter08` sang `Chapter09` và điều chỉnh nội dung text HTML.
3. Tạo file `questions.js` với 30 câu hỏi chi tiết về Chương 9.
4. Cập nhật `Chapters/chapter_09.md` bằng cách chèn thẻ `iframe` của quiz vào cuối phần "Tiếng Việt", ngay trước thẻ `<!-- tabs:end -->`.
5. Kiểm tra tính toàn vẹn (syntax lỗi, lỗi logic nếu có).
