# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter08_4th.tex`) cho **Chương 08: Logic Bậc nhất (First-Order Logic)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch08_4th.md`.

## User Review Required

> [!WARNING]
> Chương 08 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_08.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **8.1 Hạn chế của Logic Mệnh đề (10 phút)**
   - Tại sao cần biểu diễn đối tượng, quan hệ thay vì chỉ sự kiện?
2. **8.2 Cú pháp và Ngữ nghĩa của Logic Bậc nhất (30 phút)**
   - Đối tượng (Objects), Quan hệ (Relations), Hàm (Functions). Lượng từ (Quantifiers: For All, Exists).
3. **8.3 Sử dụng Logic Bậc nhất (25 phút)**
   - Các ví dụ biểu diễn: Quan hệ gia đình, Tập hợp, Danh sách.
4. **8.4 Kỹ nghệ Tri thức trong FOL (25 phút)**
   - Quy trình Kỹ nghệ Tri thức (Knowledge Engineering).

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter08_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 08.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_08.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch08_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 08.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter08_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
