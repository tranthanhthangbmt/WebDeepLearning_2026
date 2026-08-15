import os
import re

chapters_dir = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Học sâu_UDA\WebDeepLearning_2026\Chapters"
out_dir = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Học sâu_UDA\WebDeepLearning_2026\depPlan"

template = """# Lập kế hoạch: Thêm Tab Bài tập Trắc nghiệm (Interactive Quiz) cho Chương {ch_num}

Dựa trên yêu cầu, hệ thống bài tập trắc nghiệm Chương {ch_num} ("{ch_title}") sẽ được xây dựng gồm tối thiểu 30 câu hỏi. Toàn bộ nội dung kiến thức để soạn câu hỏi sẽ được trích xuất **chính xác và trực tiếp** từ tài liệu Tiếng Việt của Chương {ch_num} như sau:

## Nguồn tài liệu tham khảo chính
1. **Nội dung lý thuyết Tiếng Việt:** `Chapters/chapter_{ch_num_str}.md` (Tab "Tiếng Việt")

## Phạm vi kiến thức bao phủ (từ nguồn trên)
{topics_str}

## Proposed Changes

Tôi sẽ tạo một trang HTML chứa tối thiểu 30 câu hỏi trắc nghiệm và nhúng nó vào file `Chapters/chapter_{ch_num_str}.md`.

### Khởi tạo thư mục và file Quiz
#### [NEW] `quizzes/Chapter{ch_num_str}/index.html`
- **Thiết kế giao diện:** Tái sử dụng form giao diện, màu sắc, và cấu trúc điều khiển (HTML/CSS/JS) chuẩn như đã áp dụng cho các học phần khác (như môn Máy học) để đảm bảo tính nhất quán và chuyên nghiệp.
- **Biên soạn câu hỏi:** Dựa vào nội dung `Chapters/chapter_{ch_num_str}.md`, sinh tối thiểu 30 câu hỏi bám sát các mục lý thuyết kể trên. Đảm bảo đa dạng các loại câu hỏi (gồm: Trắc nghiệm đa lựa chọn - MCQ, Ghép nối - Matching, Sắp xếp thứ tự - Sorting, Điền từ vào chỗ trống/Kéo thả - Drag & Drop). Đồng thời, mỗi câu hỏi phải được phân loại và ghi rõ mức độ khó (Dễ, Trung bình, Khó).

### Tiêu chuẩn tối ưu hóa MCQ (Chuẩn MIT)
Trong quá trình biên soạn, đặc biệt là các câu hỏi đa lựa chọn (MCQ), cần phải tuân thủ nghiêm ngặt các quy tắc sau để đảm bảo chất lượng bài thi và tránh tình trạng sinh viên đoán được đáp án bằng mẹo:
1. **Độ dài cân bằng:** Chiều dài của đáp án đúng không được dài vượt quá **1.5 lần** chiều dài trung bình của các đáp án sai (Distractors).
2. **Cắt tỉa ngữ pháp thông minh (Smart NLP Truncation):** Tuyệt đối **không** sử dụng dấu ba chấm (`...`) để rút gọn câu. Nếu đáp án đúng quá dài, hãy cắt câu tại các ranh giới ngữ pháp tự nhiên (như trước các từ nối `vì`, `thay vì`, `nhưng`, `do đó`, `giúp`, `để`, `nghĩa là`, `trong đó` hoặc các dấu câu `,`, `:`, `;`, `-`). **Không bao giờ** cắt ở giữa một cặp ngoặc đơn `( )` để bảo toàn tính toàn vẹn ngữ nghĩa.
3. **Giữ nguyên kiến thức trong Giải thích:** Phần nội dung dài dòng bị cắt đi khỏi đáp án đúng phải được di chuyển toàn bộ, trơn tru xuống phần **Giải thích (Explanation)** để hệ thống hiển thị sau khi sinh viên trả lời xong.

### Cập nhật File Markdown
#### [MODIFY] `Chapters/chapter_{ch_num_str}.md`
Thêm tab mới vào cuối file, ngay trước `<!-- tabs:end -->`:

```markdown
#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter{ch_num_str}/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>
```

## Verification Plan
- Viết nội dung mã tạo câu hỏi và giao diện `quizzes/Chapter{ch_num_str}/index.html`.
- Mở và cập nhật file `Chapters/chapter_{ch_num_str}.md`.
- Tải lại trang web chính trên trình duyệt tại đường dẫn `/#/Chapters/chapter_{ch_num_str}` và chuyển sang tab **Bài tập Trắc nghiệm**.
- Kiểm tra tính năng tương tác (chọn đáp án, chuyển câu, nộp bài, xem kết quả đúng/sai).
"""

for i in range(2, 21):
    ch_num_str = f"{i:02d}"
    file_path = os.path.join(chapters_dir, f"chapter_{ch_num_str}.md")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find Vietnamese section
    vi_marker = "#### **Tiếng Việt"
    if vi_marker in content:
        vi_part = content.split(vi_marker)[1]
    else:
        # Fallback if no marker found
        vi_part = content
        
    # Extract Chapter Title
    ch_title = ""
    # Look for '# Chương'
    title_match = re.search(r"^#\s+Chương\s+\d+:\s*(.+)$", vi_part, re.MULTILINE)
    if title_match:
        ch_title = title_match.group(1).strip()
    else:
        # Fallback to English title if Vietnamese title not found
        title_match_en = re.search(r"^#\s+Chapter\s+\d+:\s*(.+)$", content, re.MULTILINE)
        if title_match_en:
            ch_title = title_match_en.group(1).strip()
            
    # Extract subheadings
    subheadings = []
    # Find all lines starting with '## ' (not in code blocks, just roughly)
    for line in vi_part.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            heading = line[3:].strip()
            if heading and not heading.startswith('<!--'): # Ignore comments
                subheadings.append(heading)
                
    topics_str = ""
    if subheadings:
        for idx, heading in enumerate(subheadings, 1):
            topics_str += f"{idx}. **{heading}**\n"
    else:
        topics_str = "1. **Kiến thức tổng hợp của chương**\n"
        
    plan_content = template.format(
        ch_num=i,
        ch_title=ch_title,
        ch_num_str=ch_num_str,
        topics_str=topics_str.strip()
    )
    
    out_file = os.path.join(out_dir, f"createQuizChapter{ch_num_str}.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(plan_content)
        
    print(f"Created {out_file}")

print("Done generating plans.")
