# Kế hoạch phát triển: Bộ câu hỏi trắc nghiệm Chương 14 (Phân loại văn bản)

## 1. Mục tiêu
- Xây dựng 30 câu hỏi trắc nghiệm tương tác cho Chương 14 (Phân loại văn bản).
- Giúp sinh viên ôn tập và kiểm tra kiến thức về NLP, tiền xử lý văn bản (chuẩn hóa, mã hóa), các phương pháp biểu diễn văn bản (Bag-of-words, N-gram), và mô hình trình tự (LSTM, Word Embeddings).
- Tích hợp bài trắc nghiệm vào cuối tài liệu `Chapters/chapter_14.md`.

## 2. Cấu trúc câu hỏi (30 câu)
- **15 câu Trắc nghiệm nhiều lựa chọn (MCQ):** Kiểm tra kiến thức về các khái niệm cơ bản trong NLP, các bước tiền xử lý, tf-idf, word embeddings. (Đáp án đúng phải được cân bằng chiều dài, không quá dài hoặc quá nổi bật. Các giải thích chi tiết đưa vào thuộc tính `explanation`).
- **5 câu Điền từ vào chỗ trống (Fill-in-the-blank):** Điền các thuật ngữ tiếng Anh hoặc tiếng Việt chuẩn như "tokenization", "embedding", "bag-of-words", "LSTM", "n-gram".
- **5 câu Ghép nối (Matching):** Ghép nối các khái niệm về xử lý ngôn ngữ tự nhiên, ví dụ ghép phương pháp biểu diễn văn bản với đặc điểm của nó.
- **5 câu Sắp xếp thứ tự (Sorting):** Sắp xếp các bước chuẩn hóa và tiền xử lý văn bản, hoặc quá trình xây dựng từ vựng.

## 3. Các bước thực hiện
1. **Tạo thư mục và file:**
   - Tạo thư mục `quizzes/Chapter14/`.
   - Copy các file `index.html`, `style.css`, `script.js` từ `quizzes/Chapter13/` sang `quizzes/Chapter14/`.
   - Cập nhật tiêu đề trong `index.html` thành "Bài tập Trắc nghiệm Chương 14".
2. **Soạn thảo dữ liệu câu hỏi:**
   - Tạo file `questions.js` trong `quizzes/Chapter14/` chứa mảng cấu trúc JSON export mặc định.
   - Viết 30 câu hỏi theo tỉ lệ cấu trúc ở trên.
3. **Tích hợp vào tài liệu:**
   - Mở file `Chapters/chapter_14.md`.
   - Tìm đến vị trí cuối file (trước `<!-- tabs:end -->`).
   - Chèn tab Bài tập Trắc nghiệm chứa thẻ `iframe` nhúng tới `quizzes/Chapter14/index.html`.

## 4. Yêu cầu kỹ thuật & UX/UI
- Đảm bảo tính nhất quán về giao diện, sử dụng phong cách đẹp, có phản hồi trực quan (màu xanh cho đúng, đỏ cho sai).
- Có thanh điều hướng `[< Câu trước ]`, `[ Kiểm tra ]`, `[ Câu sau > ]`.
- Câu hỏi MCQ tuân thủ nguyên tắc MIT (đáp án đúng không bất thường về độ dài).
- File `questions.js` bắt buộc phải có câu lệnh `export default questions;`.
