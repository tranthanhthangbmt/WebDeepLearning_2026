# Lập kế hoạch: Thêm Tab Bài tập Trắc nghiệm (Interactive Quiz) cho Chương 13

Dựa trên yêu cầu, hệ thống bài tập trắc nghiệm Chương 13 ("Dự báo chuỗi thời gian") sẽ được xây dựng gồm tối thiểu 30 câu hỏi. Toàn bộ nội dung kiến thức để soạn câu hỏi sẽ được trích xuất **chính xác và trực tiếp** từ tài liệu Tiếng Việt của Chương 13 như sau:

## Nguồn tài liệu tham khảo chính
1. **Nội dung lý thuyết Tiếng Việt:** `Chapters/chapter_13.md` (Tab "Tiếng Việt")

## Phạm vi kiến thức bao phủ (từ nguồn trên)
1. **Các loại nhiệm vụ theo chuỗi thời gian khác nhau**
2. **Ví dụ dự báo nhiệt độ**
3. **Mạng lưới thần kinh tái phát**
4. **Đi xa hơn nữa**
5. **Bản tóm tắt**

## Proposed Changes

Tôi sẽ tạo một trang HTML chứa tối thiểu 30 câu hỏi trắc nghiệm và nhúng nó vào file `Chapters/chapter_13.md`.

### Khởi tạo thư mục và file Quiz
#### [NEW] `quizzes/Chapter13/index.html`
- **Thiết kế giao diện:** Tái sử dụng form giao diện, màu sắc, và cấu trúc điều khiển (HTML/CSS/JS) chuẩn như đã áp dụng cho các học phần khác (như môn Máy học) để đảm bảo tính nhất quán và chuyên nghiệp.
- **Biên soạn câu hỏi:** Dựa vào nội dung `Chapters/chapter_13.md`, sinh tối thiểu 30 câu hỏi bám sát các mục lý thuyết kể trên. Đảm bảo đa dạng các loại câu hỏi (gồm: Trắc nghiệm đa lựa chọn - MCQ, Ghép nối - Matching, Sắp xếp thứ tự - Sorting, Điền từ vào chỗ trống/Kéo thả - Drag & Drop). Đồng thời, mỗi câu hỏi phải được phân loại và ghi rõ mức độ khó (Dễ, Trung bình, Khó).

### Tiêu chuẩn tối ưu hóa MCQ (Chuẩn MIT)
Trong quá trình biên soạn, đặc biệt là các câu hỏi đa lựa chọn (MCQ), cần phải tuân thủ nghiêm ngặt các quy tắc sau để đảm bảo chất lượng bài thi và tránh tình trạng sinh viên đoán được đáp án bằng mẹo:
1. **Độ dài cân bằng:** Chiều dài của đáp án đúng không được dài vượt quá **1.5 lần** chiều dài trung bình của các đáp án sai (Distractors).
2. **Cắt tỉa ngữ pháp thông minh (Smart NLP Truncation):** Tuyệt đối **không** sử dụng dấu ba chấm (`...`) để rút gọn câu. Nếu đáp án đúng quá dài, hãy cắt câu tại các ranh giới ngữ pháp tự nhiên (như trước các từ nối `vì`, `thay vì`, `nhưng`, `do đó`, `giúp`, `để`, `nghĩa là`, `trong đó` hoặc các dấu câu `,`, `:`, `;`, `-`). **Không bao giờ** cắt ở giữa một cặp ngoặc đơn `( )` để bảo toàn tính toàn vẹn ngữ nghĩa.
3. **Giữ nguyên kiến thức trong Giải thích:** Phần nội dung dài dòng bị cắt đi khỏi đáp án đúng phải được di chuyển toàn bộ, trơn tru xuống phần **Giải thích (Explanation)** để hệ thống hiển thị sau khi sinh viên trả lời xong.

### Cập nhật File Markdown
#### [MODIFY] `Chapters/chapter_13.md`
Thêm tab mới vào cuối file, ngay trước `<!-- tabs:end -->`:

```markdown
#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter13/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>
```

## Verification Plan
- Viết nội dung mã tạo câu hỏi và giao diện `quizzes/Chapter13/index.html`.
- Mở và cập nhật file `Chapters/chapter_13.md`.
- Tải lại trang web chính trên trình duyệt tại đường dẫn `/#/Chapters/chapter_13` và chuyển sang tab **Bài tập Trắc nghiệm**.
- Kiểm tra tính năng tương tác (chọn đáp án, chuyển câu, nộp bài, xem kết quả đúng/sai).
