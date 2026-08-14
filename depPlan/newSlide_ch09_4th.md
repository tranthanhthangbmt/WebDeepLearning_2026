# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter09_4th.tex`) cho **Chương 09: Suy diễn trong Logic Bậc nhất (Inference in First-Order Logic)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch09_4th.md`.

## User Review Required

> [!WARNING]
> Chương 09 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_09.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **9.1 Tri thức mệnh đề và Bậc nhất (15 phút)**
   - Loại bỏ lượng từ tồn tại (Skolemization), Đại diện toàn cục.
2. **9.2 Hợp nhất và Lược đồ nâng (Unification and Lifting) (20 phút)**
   - Quy tắc hợp nhất các biến logic (Unification algorithm).
3. **9.3 Liên kết Thuận (Forward Chaining) (20 phút)**
   - Suy luận diễn dịch theo hướng dữ liệu.
4. **9.4 Liên kết Ngược (Backward Chaining) (20 phút)**
   - Suy luận theo hướng mục tiêu (Goal-directed), Ứng dụng trong Prolog.
5. **9.5 Phân giải FOL (Resolution) (15 phút)**
   - Đưa về dạng chuẩn tắc (CNF) trong FOL.

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter09_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 09.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_09.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch09_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 09.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter09_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
