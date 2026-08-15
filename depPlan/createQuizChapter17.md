# Kế hoạch phát triển: Bộ câu hỏi trắc nghiệm Chương 17 (Tạo hình ảnh)

## 1. Mục tiêu
- Xây dựng 30 câu hỏi trắc nghiệm tương tác cho Chương 17 (Tạo hình ảnh - Image Generation).
- Giúp sinh viên ôn tập kiến thức về Bộ mã hóa tự động biến thiên (VAE), Không gian tiềm ẩn (Latent space), Mô hình khuếch tán (Diffusion Models), U-Net, và Mô hình Text-to-Image (Stable Diffusion).
- Tích hợp bài trắc nghiệm vào cuối tài liệu `Chapters/chapter_17.md`.

## 2. Cấu trúc câu hỏi (30 câu)
- **15 câu Trắc nghiệm nhiều lựa chọn (MCQ):** Kiểm tra kiến thức lý thuyết về VAE (z_mean, z_log_var, KL loss), mô hình khuếch tán (khuếch tán ngược, lịch trình cosine), Text-to-Image (Negative prompt, CLIP embedding). Các đáp án đúng phải được đảm bảo không dài bất thường để tránh làm lộ kết quả.
- **5 câu Điền từ vào chỗ trống (Fill-in-the-blank):** Điền các thuật ngữ như "VAE", "U-Net", "Khuếch tán" (Diffusion), "Tiềm ẩn" (Latent).
- **5 câu Ghép nối (Matching):** Ghép nối các mô-đun của VAE (Encoder, Decoder, Sampler) hoặc các thành phần của hàm Loss (Reconstruction Loss, KL Loss).
- **5 câu Sắp xếp thứ tự (Sorting):** Sắp xếp thứ tự các bước trong quá trình sinh ảnh của VAE, quá trình khuếch tán ngược (Reverse Diffusion), cấu trúc của mạng U-Net (Downsampling, Middle, Upsampling).

## 3. Các bước thực hiện
1. **Tạo thư mục và file:**
   - Tạo thư mục `quizzes/Chapter17/`.
   - Copy các file `index.html`, `style.css`, `script.js` từ thư mục `quizzes/Chapter16/` (hoặc 15) sang.
   - Sửa tiêu đề và nội dung UI trong `index.html` thành "Chương 17".
2. **Soạn thảo dữ liệu câu hỏi:**
   - Tạo file `questions.js` trong `quizzes/Chapter17/` và biên soạn 30 câu hỏi theo đúng chủ đề của chương.
   - Chú ý tối ưu độ dài của các câu trả lời trắc nghiệm MCQ.
3. **Tích hợp vào tài liệu:**
   - Chèn thẻ `<iframe>` trỏ đến `quizzes/Chapter17/index.html` vào cuối file `Chapters/chapter_17.md` (trước `<!-- tabs:end -->`).

## 4. Yêu cầu kỹ thuật & UX/UI
- Giao diện thân thiện, tương tự các chương trước.
- Logic tính điểm, tiến trình làm bài, chức năng làm lại đều phải hoạt động mượt mà.
- File `questions.js` phải được xuất (export default) đúng chuẩn ES6.
