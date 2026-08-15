# Lập kế hoạch: Thêm Tab Bài tập Trắc nghiệm (Interactive Quiz) cho Chương 1

Dựa trên yêu cầu, hệ thống bài tập trắc nghiệm Chương 1 ("Bức tranh tổng quan về học máy") sẽ được xây dựng gồm tối thiểu 30 câu hỏi. Toàn bộ nội dung kiến thức để soạn câu hỏi sẽ được trích xuất **chính xác và trực tiếp** từ tài liệu Tiếng Việt của Chương 1 như sau:

## Nguồn tài liệu tham khảo chính
1. **Nội dung lý thuyết Tiếng Việt:** `docs/chuong_01.md` (Tab "Lý thuyết")

## Phạm vi kiến thức bao phủ (từ nguồn trên)
1. **Định nghĩa Học máy:** Khái niệm, sự khác biệt so với phương pháp lập trình truyền thống và ưu điểm khi giải quyết vấn đề phức tạp.
2. **Ví dụ ứng dụng của Học máy:** Phân loại hình ảnh, xử lý ngôn ngữ tự nhiên (NLP), hồi quy (dự đoán), phát hiện bất thường, phân cụm, hệ thống đề xuất, học tăng cường...
3. **Phân loại hệ thống ML theo mức độ giám sát:** 
   - Học có giám sát (Supervised learning): Phân loại, Hồi quy.
   - Học không giám sát (Unsupervised learning): Phân cụm, giảm chiều, phát hiện bất thường, học luật kết hợp.
   - Học bán giám sát (Semi-supervised learning) và Học tự giám sát (Self-supervised learning).
   - Học tăng cường (Reinforcement learning).
4. **Phân loại hệ thống ML theo cách học liên tục:** 
   - Học theo lô (Batch learning / Offline learning).
   - Học trực tuyến (Online learning), tốc độ học (learning rate), và giải quyết các luồng dữ liệu khổng lồ (out-of-core learning).
5. **Phân loại hệ thống ML theo cách tổng quát hóa:**
   - Học dựa trên thực thể (Instance-based learning).
   - Học dựa trên mô hình (Model-based learning).
6. **Quy trình làm việc ML:** Dữ liệu huấn luyện, đặc trưng đầu vào, dự đoán kết quả và cải thiện mô hình.

## Proposed Changes

Tôi sẽ tạo một trang HTML chứa tối thiểu 30 câu hỏi trắc nghiệm và nhúng nó vào file `docs/chuong_01.md`.

### Khởi tạo thư mục và file Quiz
#### [NEW] `quizzes/Chapter01/index.html`
- **Thiết kế giao diện:** Tái sử dụng form giao diện, màu sắc, và cấu trúc điều khiển (HTML/CSS/JS) chuẩn như đã áp dụng cho các học phần khác (ví dụ: AI Kế toán) để đảm bảo tính nhất quán và chuyên nghiệp.
- **Biên soạn câu hỏi:** Dựa vào nội dung `docs/chuong_01.md`, sinh tối thiểu 30 câu hỏi bám sát các mục lý thuyết kể trên. Đảm bảo đa dạng các loại câu hỏi như môn AI cho Kế toán (gồm: Trắc nghiệm đa lựa chọn - MCQ, Ghép nối - Matching, Sắp xếp thứ tự - Sorting, Điền từ vào chỗ trống/Kéo thả - Drag & Drop). Đồng thời, mỗi câu hỏi phải được phân loại và ghi rõ mức độ khó (Dễ, Trung bình, Khó).

### Tiêu chuẩn tối ưu hóa MCQ (Chuẩn MIT)
Trong quá trình biên soạn, đặc biệt là các câu hỏi đa lựa chọn (MCQ), cần phải tuân thủ nghiêm ngặt các quy tắc sau để đảm bảo chất lượng bài thi và tránh tình trạng sinh viên đoán được đáp án bằng mẹo:
1. **Độ dài cân bằng:** Chiều dài của đáp án đúng không được dài vượt quá **1.5 lần** chiều dài trung bình của các đáp án sai (Distractors).
2. **Cắt tỉa ngữ pháp thông minh (Smart NLP Truncation):** Tuyệt đối **không** sử dụng dấu ba chấm (`...`) để rút gọn câu. Nếu đáp án đúng quá dài, hãy cắt câu tại các ranh giới ngữ pháp tự nhiên (như trước các từ nối `vì`, `thay vì`, `nhưng`, `do đó`, `giúp`, `để`, `nghĩa là`, `trong đó` hoặc các dấu câu `,`, `:`, `;`, `-`). **Không bao giờ** cắt ở giữa một cặp ngoặc đơn `( )` để bảo toàn tính toàn vẹn ngữ nghĩa.
3. **Giữ nguyên kiến thức trong Giải thích:** Phần nội dung dài dòng bị cắt đi khỏi đáp án đúng phải được di chuyển toàn bộ, trơn tru xuống phần **Giải thích (Explanation)** để hệ thống hiển thị sau khi sinh viên trả lời xong.

### Cập nhật File Markdown
#### [MODIFY] `docs/chuong_01.md`
Thêm tab mới vào cuối file, ngay trước `<!-- tabs:end -->`:

```markdown
#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter01/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>
```

## Verification Plan
- Viết nội dung mã tạo câu hỏi và giao diện `quizzes/Chapter01/index.html`.
- Mở và cập nhật file `docs/chuong_01.md`.
- Tải lại trang web chính trên trình duyệt tại đường dẫn `/#/docs/chuong_01` và chuyển sang tab **Bài tập Trắc nghiệm**.
- Kiểm tra tính năng tương tác (chọn đáp án, chuyển câu, nộp bài, xem kết quả đúng/sai).
