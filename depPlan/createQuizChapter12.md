# Kế hoạch tạo bài tập trắc nghiệm (Interactive Quiz) cho Chương 12

## 1. Mục tiêu
- Xây dựng bộ câu hỏi tương tác kiểm tra kiến thức về **Chương 12: Phát hiện đối tượng (Object Detection)**.
- Tiếp tục tái sử dụng bộ khung UI và phong cách thiết kế quen thuộc để mang lại trải nghiệm nhất quán cho người dùng.

## 2. Chủ đề câu hỏi (30 câu)
1. **Khái niệm cơ bản & Ứng dụng:**
   - So sánh Object Detection vs Image Segmentation (Tốc độ, nhu cầu nhãn pixel vs hộp giới hạn, bài toán đếm/theo dõi).
   - Ứng dụng phổ biến (Counting, Tracking, Cropping).
2. **Máy dò hai giai đoạn (Two-stage Detectors):**
   - Dòng R-CNN (Region-based CNN).
   - Giai đoạn 1: Đề xuất khu vực (Region Proposals) bằng Selective Search hoặc RPN.
   - Giai đoạn 2: Phân loại từng đề xuất. Đặc điểm: Chính xác nhưng cực kỳ chậm, không hợp thời gian thực.
3. **Máy dò một giai đoạn (Single-stage Detectors):**
   - YOLO (You Only Look Once), SSD, RetinaNet.
   - Trực tiếp dự đoán hộp giới hạn (Bounding boxes) và Nhãn lớp (Class labels) cùng một lúc. Đặc điểm: Nhanh, phù hợp real-time.
4. **Mô hình YOLO & Bộ dữ liệu COCO:**
   - Dữ liệu COCO (Common Objects in Context): 91 lớp.
   - Định dạng hộp YOLO `(x, y, w, h, confidence)`.
   - Grid cell, backbone ResNet (strided convolutions thay vì max pooling).
   - Hàm Loss của YOLO: `sparse_categorical_crossentropy` cho class, và Box loss có cơ chế trừng phạt (scale factor 5x cho box, 0.5 cho confidence khi ô trống).
5. **Mô hình RetinaNet & Feature Pyramid Network (FPN):**
   - Yếu điểm của YOLO: Khó phát hiện vật thể nhỏ.
   - Giải pháp của RetinaNet (FPN): Kết hợp đặc trưng độ phân giải cao (ít ngữ nghĩa) từ các lớp đầu ConvNet với đặc trưng độ phân giải thấp (nhiều ngữ nghĩa) từ các lớp cuối.

## 3. Cấu trúc file (`quizzes/Chapter12`)
- **`index.html`**: File giao diện nhúng. Tiêu đề: Chương 12.
- **`style.css`, `script.js`**: Re-use từ Chapter 11.
- **`questions.js`**: 30 câu hỏi (25 MCQ, 2 Điền từ, 2 Ghép nối, 1 Sắp xếp).
  - Chiều dài các đáp án đúng và sai trong MCQ phải được thiết kế gần bằng nhau.
  - Export module bằng `export default questions;`.

## 4. Các bước thực thi
1. Tạo thư mục `quizzes/Chapter12`.
2. Copy giao diện từ `Chapter11` qua và chỉnh sửa text `index.html`.
3. Sinh 30 câu hỏi chuyên sâu đưa vào `questions.js`.
4. Nhúng quiz vào cuối file `Chapters/chapter_12.md`.
5. Hoàn tất và chờ đánh giá.
