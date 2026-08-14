# Kế hoạch dịch chi tiết Chương 22 (Deep Learning)

Dựa trên thành công của quá trình dịch Chương 21, tôi đề xuất lặp lại quy trình "chia nhỏ để không bị tóm tắt" đối với Chương 22.

## Background Context
File `Chapter_22_Deep Learning.pdf` có tổng cộng 39 trang, dài hơn so với Chương 21. Việc đưa cho AI toàn bộ một lượng lớn văn bản sẽ khiến mô hình tự động tóm tắt nội dung, làm mất đi các chi tiết quan trọng và công thức toán học. Để giải quyết triệt để, chúng ta cần chia nhỏ file này.

## User Review Required
Bạn vui lòng kiểm tra kế hoạch dưới đây. Nếu bạn đồng ý, tôi sẽ tự động tiến hành từng bước. Với 39 trang, tôi đề xuất chia thành **13 phần** (mỗi phần 3 trang) để đảm bảo chất lượng dịch thuật tốt nhất.

> [!IMPORTANT]
> Hãy nhấp "Proceed" để tôi bắt đầu thực hiện kế hoạch nếu bạn đồng ý.

## Proposed Changes

### 1. Phân tách Dữ liệu (Extraction)
- Chạy script Python (`extract_pages_22.py`) để đọc `TaiLieu/ebooks_Chapters_Vi3/Chapter_22_Deep Learning.pdf`.
- Trích xuất văn bản và chia thành 13 file từ `part1_22.txt` đến `part13_22.txt` (mỗi file tương ứng với 3 trang).

### 2. Dịch thuật Chi tiết (Translation)
Tiến hành dịch lần lượt từng phần và lưu kết quả vào thư mục `TaiLieu/ebooks_Chapters_Vi3/Chapter_22_Deep Learning/` dưới dạng các file HTML (`part1_22.html` -> `part13_22.html`). Các yêu cầu bắt buộc:
- Giữ nguyên định dạng gốc của văn bản.
- Giữ thuật ngữ tiếng Anh gốc trong ngoặc đơn (ví dụ: học sâu (deep learning)).
- Đảm bảo giữ trọn vẹn và render đúng các công thức toán học bằng chuẩn `MathJax`.
- Tuyệt đối không tóm tắt, không bỏ sót bất kỳ dòng nào.

### 3. Gộp và Tích hợp (Integration)
- Gộp 13 file HTML lại thành một file hoàn chỉnh: `chapter_22_vi.html`.
- Xóa bỏ hoặc không sử dụng `polyfill.io` trong HTML template (để tránh lỗi trắng màn hình).
- Thay thế bằng `<script>` tải `MathJax` từ `jsdelivr`.

### 4. Cập nhật Markdown và Xóa Cache
- Chỉnh sửa file `chapters/chapter_22_deep_learning.md` (hoặc tên tương ứng nếu có khác biệt đôi chút) để thay đổi thẻ `<iframe>` trỏ đến file `chapter_22_vi.html` mới.
- Thêm tham số `?v=2` vào cuối đường dẫn (ví dụ: `chapter_22_vi.html?v=2`) để bắt trình duyệt của bạn tải lại bản mới nhất.

## Verification Plan

### Automated Tests
- Chạy thử Python script đảm bảo trích xuất đủ 13 file `partX_22.txt`.
- Đảm bảo file cuối cùng `chapter_22_vi.html` được tạo thành công.

### Manual Verification
- Bạn sẽ tải lại web để kiểm tra xem chương 22 đã hiển thị đủ các phần hay chưa, công thức toán học có hoạt động tốt không.
