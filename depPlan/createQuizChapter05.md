# Lập kế hoạch: Thêm Tab Bài tập Trắc nghiệm (Interactive Quiz) cho Chương 5

Dựa trên yêu cầu, hệ thống bài tập trắc nghiệm Chương 5 ("Nguyên tắc cơ bản của học máy") sẽ được xây dựng gồm tối thiểu 30 câu hỏi. Toàn bộ nội dung kiến thức để soạn câu hỏi sẽ được trích xuất **chính xác và trực tiếp** từ tài liệu Tiếng Việt của Chương 5 như sau:

## Nguồn tài liệu tham khảo chính
1. **Nội dung lý thuyết Tiếng Việt:** `Chapters/chapter_05.md` (Tab "Tiếng Việt")

## Phạm vi kiến thức bao phủ (từ nguồn trên)
1. **Khái quát hóa: Mục tiêu của học máy**
2. **Đánh giá các mô hình học máy**
3. **Cải thiện sự phù hợp của mô hình**
4. **Cải thiện khái quát hóa**
5. **Bản tóm tắt**

## Proposed Changes

Tôi sẽ tạo một trang HTML chứa tối thiểu 30 câu hỏi trắc nghiệm và nhúng nó vào file `Chapters/chapter_05.md`.

### Khởi tạo thư mục và file Quiz
#### [NEW] `quizzes/Chapter05/index.html`
- **Thiết kế giao diện:** Tái sử dụng form giao diện, màu sắc, và cấu trúc điều khiển (HTML/CSS/JS) chuẩn như đã áp dụng cho các học phần khác (như môn Máy học) để đảm bảo tính nhất quán và chuyên nghiệp.
- **Biên soạn câu hỏi:** Dựa vào nội dung `Chapters/chapter_05.md`, sinh tối thiểu 30 câu hỏi bám sát các mục lý thuyết kể trên. Đảm bảo đa dạng các loại câu hỏi (gồm: Trắc nghiệm đa lựa chọn - MCQ, Ghép nối - Matching, Sắp xếp thứ tự - Sorting, Điền từ vào chỗ trống/Kéo thả - Drag & Drop). Đồng thời, mỗi câu hỏi phải được phân loại và ghi rõ mức độ khó (Dễ, Trung bình, Khó).

### Tiêu chuẩn UX/UI Điều hướng & Học tập tương tác
Kế thừa thành công từ các chương trước, hệ thống Quiz Chương 5 phải đảm bảo:
1. **Thanh điều hướng 3 nút chuẩn mực:** 
   - `[< Câu trước ]`: Đặt bên trái, hiển thị từ câu thứ 2.
   - `[ Kiểm tra ]`: Nằm chính giữa, dùng cho các dạng câu hỏi cần thao tác nhiều bước (Điền từ, Sắp xếp, Ghép nối).
   - `[ Câu sau > ]`: Đặt bên phải, luôn hiển thị để sinh viên có thể bỏ qua (skip) câu hỏi khó.
2. **Practice Mode cho dạng Trắc nghiệm (MCQ):**
   - Ẩn hoàn toàn nút `Kiểm tra`.
   - Chấm điểm và đổi màu (Đỏ = Sai, Xanh = Đúng) ngay lập tức khi người học click vào đáp án.
   - **Được phép đoán lại:** Nếu chọn sai, người học vẫn được chọn tiếp đến khi đúng.
   - Bảng **💡 Giải thích** chỉ hiển thị khi chọn đúng đáp án. Điểm số chỉ được ghi nhận nếu chọn đúng ngay từ lần đầu tiên.

### Tiêu chuẩn tối ưu hóa MCQ (Chuẩn MIT)
Trong quá trình biên soạn, đặc biệt là các câu hỏi đa lựa chọn (MCQ), cần phải tuân thủ nghiêm ngặt các quy tắc sau để đảm bảo chất lượng bài thi và tránh tình trạng sinh viên đoán được đáp án bằng mẹo:
1. **Độ dài cân bằng tuyệt đối:** Chiều dài của đáp án đúng **TUYỆT ĐỐI KHÔNG** được dài bất thường hoặc khác biệt rõ rệt so với các đáp án sai (Distractors). Phải đảm bảo người làm bài không thể dùng mẹo "câu dài nhất là câu đúng".
2. **Cắt tỉa ngữ pháp thông minh (Smart NLP Truncation):** Tuyệt đối **không** sử dụng dấu ba chấm (`...`) để rút gọn câu. Nếu đáp án đúng quá dài, hãy cắt câu tại các ranh giới ngữ pháp tự nhiên.
3. **Giữ nguyên kiến thức trong Giải thích:** Phần nội dung dài dòng bị cắt đi khỏi đáp án đúng phải được di chuyển toàn bộ xuống phần **Giải thích (Explanation)** để hệ thống hiển thị sau khi sinh viên trả lời đúng.

### Cập nhật File Markdown
#### [MODIFY] `Chapters/chapter_05.md`
Thêm tab mới vào cuối file, ngay trước `<!-- tabs:end -->`:

```markdown
#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter05/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>
```

## Verification Plan
- Viết nội dung mã tạo câu hỏi và giao diện `quizzes/Chapter05/index.html`.
- Mở và cập nhật file `Chapters/chapter_05.md`.
- Tải lại trang web chính trên trình duyệt tại đường dẫn `/#/Chapters/chapter_05` và chuyển sang tab **Bài tập Trắc nghiệm**.
- Kiểm tra tính năng tương tác (chọn đáp án, chuyển câu, nộp bài, xem kết quả đúng/sai).
