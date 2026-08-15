# Kế hoạch phát triển: Bộ câu hỏi trắc nghiệm Chương 16 (Tạo văn bản)

## 1. Mục tiêu
- Xây dựng 30 câu hỏi trắc nghiệm tương tác cho Chương 16 (Tạo văn bản - Generative text).
- Giúp sinh viên ôn tập kiến thức về mô hình ngôn ngữ lớn (LLMs), GPT, kỹ thuật giải mã sinh văn bản (Sampling strategies), tinh chỉnh lệnh (Instruction tuning), và tinh chỉnh tiết kiệm bộ nhớ (LoRA).
- Tích hợp bài trắc nghiệm vào cuối tài liệu `Chapters/chapter_16.md`.

## 2. Cấu trúc câu hỏi (30 câu)
- **15 câu Trắc nghiệm nhiều lựa chọn (MCQ):** Kiểm tra kiến thức về LLM, quá trình tiền huấn luyện (Pretraining), Temperature, Top-k, hiện tượng Hallucination, tinh chỉnh LoRA, Decoder-only vs Encoder-Decoder, caching. Đảm bảo độ dài đáp án đúng tương đương với các đáp án nhiễu.
- **5 câu Điền từ vào chỗ trống (Fill-in-the-blank):** Điền các thuật ngữ quan trọng như "GPT", "Temperature", "LoRA", "Ảo giác".
- **5 câu Ghép nối (Matching):** Ghép nối các chiến lược lấy mẫu (Greedy, Random, Top-k) hoặc các khái niệm trong quá trình tinh chỉnh (Instruction, Response).
- **5 câu Sắp xếp thứ tự (Sorting):** Sắp xếp thứ tự các bước trong quá trình tạo (Generation loop), quá trình tinh chỉnh LoRA hoặc tiền huấn luyện mô hình ngôn ngữ.

## 3. Các bước thực hiện
1. **Tạo thư mục và file:**
   - Tạo thư mục `quizzes/Chapter16/`.
   - Copy các file `index.html`, `style.css`, `script.js` từ `quizzes/Chapter15/` sang `quizzes/Chapter16/`.
   - Cập nhật tiêu đề trong `index.html` thành "Bài tập Trắc nghiệm Chương 16".
2. **Soạn thảo dữ liệu câu hỏi:**
   - Tạo file `questions.js` trong `quizzes/Chapter16/` chứa mảng đối tượng câu hỏi với từ khóa xuất (export default).
   - Viết 30 câu hỏi tuân thủ đúng yêu cầu chất lượng (đặc biệt là yêu cầu về độ dài đáp án MCQ).
3. **Tích hợp vào tài liệu:**
   - Mở file `Chapters/chapter_16.md`.
   - Tìm đến phần cuối của file (trước thẻ đóng tab nếu có).
   - Chèn iframe tab Bài tập Trắc nghiệm nhúng vào `quizzes/Chapter16/index.html`.

## 4. Yêu cầu kỹ thuật & UX/UI
- Tính tương tác tốt, phản hồi màu sắc rõ ràng (xanh - đỏ) kèm ô giải thích sau khi trả lời.
- Giao diện có thanh điều hướng các câu.
- Cấu trúc export chuẩn ES6 module cho file `questions.js`.
