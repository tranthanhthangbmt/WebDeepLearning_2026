# Kế hoạch tạo bài tập trắc nghiệm (Interactive Quiz) cho Chương 11

## 1. Mục tiêu
- Xây dựng hệ thống câu hỏi trắc nghiệm tương tác cho **Chương 11: Phân đoạn hình ảnh (Image segmentation)**.
- Tích hợp giao diện UI tương tự như đã làm ở các chương trước vào bài học Markdown, giúp sinh viên củng cố kiến thức trực tiếp.

## 2. Chủ đề câu hỏi (30 câu)
Dựa trên nội dung của Chương 11, các câu hỏi sẽ xoay quanh các chủ đề:
1. **Các tác vụ Computer Vision cơ bản:** Phân loại ảnh (Classification), Phân đoạn ảnh (Segmentation), Phát hiện đối tượng (Object Detection).
2. **Các loại phân đoạn hình ảnh:** Semantic segmentation (Ngữ nghĩa), Instance segmentation (Phiên bản), Panoptic segmentation (Toàn cảnh).
3. **Xây dựng mô hình phân đoạn từ đầu:**
   - Cấu trúc downsampling: Tại sao dùng Conv2D với strides thay vì MaxPooling2D (bảo toàn thông tin vị trí không gian).
   - Cấu trúc upsampling: Lớp `Conv2DTranspose` (học cách lấy mẫu lên).
   - Hàm đánh giá IoU (Intersection over Union).
   - Cấu trúc mặt nạ (mask) đầu ra (VD: Tiền cảnh, Hậu cảnh, Đường viền).
4. **Mô hình Segment Anything (SAM):**
   - Sự ra đời, bộ dữ liệu khổng lồ SA-1B, tác giả Meta AI.
   - Các loại Prompts (dấu nhắc): point (điểm), box (hộp bao).
   - Kiến trúc SAM (Image Encoder, Prompt Encoder, Mask Decoder).
   - Khả năng phân đoạn zero-shot không cần huấn luyện lại trên dữ liệu mới.

## 3. Cấu trúc và Giao diện (Thư mục `quizzes/Chapter11`)
- **`index.html`**: File giao diện chính, được nhúng qua iframe vào `Chapters/chapter_11.md`. Cập nhật tiêu đề thành Chương 11.
- **`style.css`**: Dùng chung phong cách (giữ nguyên).
- **`script.js`**: Logic xử lý UI (giữ nguyên).
- **`questions.js`**: Chứa 30 câu hỏi dưới định dạng mảng JSON. 
  - Tuân thủ nguyên tắc: **Chiều dài các đáp án phải tương đương nhau**. Giải thích chi tiết cho đáp án đúng.
  - Phân bổ: 25 Multiple Choice (MCQ), 2 Matching, 1 Sorting, 2 Fill-in-the-blank.
  - Phải có dòng `export default questions;` ở cuối.

## 4. Các bước thực hiện
1. Tạo thư mục `quizzes/Chapter11`.
2. Sao chép `index.html`, `style.css`, `script.js` từ `Chapter10` sang `Chapter11` và điều chỉnh nội dung text HTML.
3. Tạo file `questions.js` với 30 câu hỏi chi tiết về Chương 11.
4. Cập nhật `Chapters/chapter_11.md` bằng cách chèn thẻ `iframe` của quiz vào cuối phần "Tiếng Việt", ngay trước thẻ `<!-- tabs:end -->`.
5. Kiểm tra tính toàn vẹn (lỗi cú pháp, logic).
