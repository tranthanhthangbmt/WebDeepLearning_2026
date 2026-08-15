# Lập kế hoạch: Thêm Tab Bài tập Trắc nghiệm (Interactive Quiz) cho Chương 8

Dựa trên yêu cầu, hệ thống bài tập trắc nghiệm Chương 8 ("Phân loại hình ảnh") sẽ được xây dựng gồm tối thiểu 30 câu hỏi. Toàn bộ nội dung kiến thức để soạn câu hỏi sẽ được trích xuất **chính xác và trực tiếp** từ tài liệu Tiếng Việt của Chương 8 như sau:

## Nguồn tài liệu tham khảo chính
1. **Nội dung lý thuyết Tiếng Việt:** `Chapters/chapter_08.md` (Tab "Tiếng Việt")

## Phạm vi kiến thức bao phủ (từ nguồn trên)
1. **ConvNets và Tích chập (Convolution):** Đặc điểm nhận diện mẫu cục bộ (local patterns), tính bất biến dịch chuyển (translation invariance) và học các phân cấp không gian.
2. **Feature Maps và Filters:** Bản đồ đặc trưng, bản đồ phản hồi, độ sâu đầu ra, và kernel tích chập.
3. **Hiệu ứng biên và Padding:** "valid" vs "same" padding.
4. **Sải bước (Strides) và MaxPooling:** Cách giảm kích thước bản đồ đặc trưng, gộp tối đa so với gộp trung bình.
5. **Huấn luyện mô hình từ đầu với dữ liệu nhỏ:** Bài toán Dogs vs. Cats, lấy mẫu con.
6. **Dataset trong TensorFlow (`tf.data.Dataset`):** Hiệu suất, batching, mapping và prefetching.
7. **Tăng cường dữ liệu (Data Augmentation):** Các lớp `RandomFlip`, `RandomRotation`, `RandomZoom` để chống Overfitting. Tầm quan trọng của việc gọi chúng trong pipeline dữ liệu thay vì trong model (nếu chạy CPU).
8. **Mô hình huấn luyện trước (Pretrained Models):** Cách tận dụng mô hình đã huấn luyện trên ImageNet (Xception). Khái niệm Feature Extraction và Fine-tuning.
9. **Trích xuất đặc trưng (Feature Extraction):** Tại sao chỉ giữ lại phần cơ sở tích chập (convolutional base) mà không giữ lại lớp phân loại. Kỹ thuật Extract fast (không Augmentation) vs. Extract end-to-end (có Augmentation).
10. **Đóng băng trọng số (Freezing):** `trainable=False`.
11. **Tinh chỉnh (Fine-tuning):** Các bước để mở băng (unfreeze) các lớp đỉnh và huấn luyện lại với learning rate nhỏ để không phá vỡ trọng số.

## Proposed Changes

Tôi sẽ tạo một trang HTML chứa tối thiểu 30 câu hỏi trắc nghiệm và nhúng nó vào file `Chapters/chapter_08.md`.

### Khởi tạo thư mục và file Quiz
#### [NEW] `quizzes/Chapter08/index.html`
- **Thiết kế giao diện:** Tái sử dụng form giao diện, màu sắc, và cấu trúc điều khiển (HTML/CSS/JS) chuẩn như đã áp dụng cho các học phần khác để đảm bảo tính nhất quán và chuyên nghiệp.
- **Biên soạn câu hỏi:** Dựa vào nội dung `Chapters/chapter_08.md`, sinh tối thiểu 30 câu hỏi bám sát các mục lý thuyết kể trên. Đảm bảo đa dạng các loại câu hỏi (gồm: Trắc nghiệm đa lựa chọn - MCQ, Ghép nối - Matching, Sắp xếp thứ tự - Sorting, Điền từ vào chỗ trống/Kéo thả - Drag & Drop). Đồng thời, mỗi câu hỏi phải được phân loại và ghi rõ mức độ khó (Dễ, Trung bình, Khó).

### Tiêu chuẩn tối ưu hóa MCQ (Chuẩn MIT)
Trong quá trình biên soạn, đặc biệt là các câu hỏi đa lựa chọn (MCQ), cần phải tuân thủ nghiêm ngặt các quy tắc sau để đảm bảo chất lượng bài thi và tránh tình trạng sinh viên đoán được đáp án bằng mẹo:
1. **Độ dài cân bằng tuyệt đối:** Chiều dài của đáp án đúng TUYỆT ĐỐI KHÔNG được dài bất thường hoặc khác biệt rõ rệt so với các đáp án sai (Distractors).
2. **Cắt tỉa ngữ pháp thông minh (Smart NLP Truncation):** Tuyệt đối **không** sử dụng dấu ba chấm (`...`) để rút gọn câu. Nếu đáp án đúng quá dài, hãy cắt câu tại các ranh giới ngữ pháp tự nhiên.
3. **Giữ nguyên kiến thức trong Giải thích:** Phần nội dung dài dòng bị cắt đi khỏi đáp án đúng phải được di chuyển toàn bộ, trơn tru xuống phần **Giải thích (Explanation)** để hệ thống hiển thị sau khi sinh viên trả lời xong.

### Cập nhật File Markdown
#### [MODIFY] `Chapters/chapter_08.md`
Thêm tab mới vào cuối file, ngay trước `<!-- tabs:end -->`:

```markdown
#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter08/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>
```

## Verification Plan
- Chạy lệnh copy thư mục giao diện để tiết kiệm thời gian.
- Cập nhật HTML để trỏ đúng tên chương.
- Tạo `questions.js` với 30 câu hỏi.
- Chèn iframe vào markdown.
- Kiểm tra lại cấu trúc.
