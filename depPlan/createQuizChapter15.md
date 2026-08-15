# Kế hoạch phát triển: Bộ câu hỏi trắc nghiệm Chương 15 (Mô hình ngôn ngữ và Transformer)

## 1. Mục tiêu
- Xây dựng 30 câu hỏi trắc nghiệm tương tác cho Chương 15 (Mô hình ngôn ngữ và Transformer).
- Giúp sinh viên ôn tập và kiểm tra kiến thức về các mô hình tạo văn bản, dịch máy (Seq2Seq, RNN Encoder-Decoder), cơ chế Attention (Chú ý) và kiến trúc Transformer.
- Tích hợp bài trắc nghiệm vào cuối tài liệu `Chapters/chapter_15.md`.

## 2. Cấu trúc câu hỏi (30 câu)
- **15 câu Trắc nghiệm nhiều lựa chọn (MCQ):** Kiểm tra kiến thức về các khái niệm cơ bản (Language Model, Seq2Seq, Encoder, Decoder, Attention, Transformer, Multi-Head Attention, Query-Key-Value). Đảm bảo tính cân bằng về chiều dài của các đáp án đúng để tránh bộc lộ mẹo làm bài.
- **5 câu Điền từ vào chỗ trống (Fill-in-the-blank):** Điền các thuật ngữ chính xác như "Encoder", "Decoder", "Attention", "Transformer", "Seq2Seq".
- **5 câu Ghép nối (Matching):** Ghép nối các thành phần của kiến trúc Transformer (ví dụ: Query, Key, Value) với ý nghĩa hoặc ví dụ cụ thể của chúng.
- **5 câu Sắp xếp thứ tự (Sorting):** Sắp xếp quá trình giải mã (decoding) một câu trong mô hình seq2seq hoặc các bước thực hiện quá trình Self-attention.

## 3. Các bước thực hiện
1. **Tạo thư mục và file:**
   - Tạo thư mục `quizzes/Chapter15/`.
   - Copy các file `index.html`, `style.css`, `script.js` từ `quizzes/Chapter14/` sang `quizzes/Chapter15/`.
   - Cập nhật tiêu đề trong `index.html` thành "Bài tập Trắc nghiệm Chương 15".
2. **Soạn thảo dữ liệu câu hỏi:**
   - Tạo file `questions.js` trong `quizzes/Chapter15/` chứa mảng đối tượng câu hỏi với từ khóa xuất (export default).
   - Viết 30 câu hỏi tuân thủ đúng yêu cầu chất lượng (đặc biệt là yêu cầu về độ dài đáp án MCQ).
3. **Tích hợp vào tài liệu:**
   - Mở file `Chapters/chapter_15.md`.
   - Tìm đến thẻ `<!-- tabs:end -->` ở phần cuối.
   - Chèn iframe tab Bài tập Trắc nghiệm nhúng vào `quizzes/Chapter15/index.html`.

## 4. Yêu cầu kỹ thuật & UX/UI
- Tính tương tác tốt, phản hồi màu sắc rõ ràng (xanh - đỏ) kèm ô giải thích sau khi trả lời.
- Giao diện có thanh điều hướng các câu.
- Cấu trúc export chuẩn ES6 module cho file `questions.js`.
