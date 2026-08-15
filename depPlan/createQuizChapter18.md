# Kế hoạch phát triển: Bộ câu hỏi trắc nghiệm Chương 18 (Những phương pháp thực hành tốt nhất cho thế giới thực)

## 1. Mục tiêu
- Xây dựng 30 câu hỏi trắc nghiệm tương tác cho Chương 18 (Best practices for the real world).
- Giúp sinh viên ôn tập kiến thức về KerasTuner (điều chỉnh siêu tham số), Ensembling (tổ hợp mô hình), Đào tạo phân tán (Multi-GPU/TPU, Data/Model parallelism), và Tối ưu hóa tính toán (Mixed-precision, Quantization).
- Tích hợp bài trắc nghiệm vào cuối tài liệu `Chapters/chapter_18.md`.

## 2. Cấu trúc câu hỏi (30 câu)
- **15 câu Trắc nghiệm nhiều lựa chọn (MCQ):** Kiểm tra về KerasTuner, overfitting tập xác thực khi dò tham số, sự đa dạng trong Ensembling, Data parallelism, Mixed-precision (float16/float32) và Quantization (int8). Các đáp án đúng phải được đảm bảo không dài bất thường để tránh làm lộ kết quả.
- **5 câu Điền từ vào chỗ trống (Fill-in-the-blank):** Điền các thuật ngữ như "siêu tham số" (hyperparameters), "tổ hợp" (ensemble), "song song dữ liệu" (data parallelism), "lượng tử hóa" (quantization).
- **5 câu Ghép nối (Matching):** Ghép nối các khái niệm trong KerasTuner (RandomSearch, BayesianOptimization), các mức độ chính xác (float32, float16, int8), các kỹ thuật phân tán (Data parallelism vs Model parallelism).
- **5 câu Sắp xếp thứ tự (Sorting):** Sắp xếp quy trình dò tìm siêu tham số, quy trình huấn luyện với Mixed Precision, quá trình lượng tử hóa int8.

## 3. Các bước thực hiện
1. **Tạo thư mục và file:**
   - Tạo thư mục `quizzes/Chapter18/`.
   - Copy các file `index.html`, `style.css`, `script.js` từ thư mục `quizzes/Chapter17/` sang.
   - Sửa tiêu đề và nội dung UI trong `index.html` thành "Chương 18".
2. **Soạn thảo dữ liệu câu hỏi:**
   - Tạo file `questions.js` trong `quizzes/Chapter18/` và biên soạn 30 câu hỏi theo đúng chủ đề của chương.
   - Chú ý tối ưu độ dài của các câu trả lời trắc nghiệm MCQ.
3. **Tích hợp vào tài liệu:**
   - Chèn thẻ `<iframe>` trỏ đến `quizzes/Chapter18/index.html` vào cuối file `Chapters/chapter_18.md` (trước `<!-- tabs:end -->`).

## 4. Yêu cầu kỹ thuật & UX/UI
- Giao diện thân thiện, tương tự các chương trước.
- Logic tính điểm, tiến trình làm bài, chức năng làm lại đều phải hoạt động mượt mà.
- File `questions.js` phải được xuất (export default) đúng chuẩn ES6.
