# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter05_4th.tex`) cho **Chương 05: Bài toán Thỏa mãn Ràng buộc (Constraint Satisfaction Problems)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch05_4th.md`.

## User Review Required

> [!WARNING]
> Chương 05 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_05.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **5.1 Định nghĩa CSP (15 phút)**
   - Biến (Variables), Miền giá trị (Domains), Ràng buộc (Constraints). Lấy ví dụ tô màu bản đồ.
2. **5.2 Suy diễn CSP: Lan truyền Ràng buộc (25 phút)**
   - Node consistency, Arc consistency (AC-3), Path consistency.
3. **5.3 Tìm kiếm Quay lui (Backtracking) cho CSP (25 phút)**
   - Lựa chọn biến (MRV, Degree heuristic), Lựa chọn giá trị (LCV), Kiểm tra trước (Forward checking).
4. **5.4 Tìm kiếm Cục bộ cho CSP (15 phút)**
   - Heuristic Min-conflicts.
5. **5.5 Cấu trúc bài toán (10 phút)**
   - Bài toán đồ thị độc lập và Cây ràng buộc.

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter05_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 05.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_05.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch05_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 05.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter05_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
