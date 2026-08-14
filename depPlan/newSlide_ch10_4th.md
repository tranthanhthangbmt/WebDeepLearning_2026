# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter10_4th.tex`) cho **Chương 10: Biểu diễn Tri thức (Knowledge Representation)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch10_4th.md`.

## User Review Required

> [!WARNING]
> Chương 10 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_10.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **10.1 Kỹ thuật Tri thức Thực thể (Ontological Engineering) (15 phút)**
   - Các mức trừu tượng, phân cấp thực thể.
2. **10.2 Phân loại và Đối tượng (Categories and Objects) (20 phút)**
   - Mạng ngữ nghĩa (Semantic Networks). Subclass, Khung (Frames).
3. **10.3 Biểu diễn Sự kiện (Events) (20 phút)**
   - Tính thời gian, hoàn cảnh, sự biến đổi theo thời gian.
4. **10.4 Thực thể niềm tin (Mental Events) (20 phút)**
   - Tri thức về tri thức, mô hình hóa suy nghĩ của tác nhân khác.
5. **10.5 Hệ thống suy diễn cho Danh mục (15 phút)**
   - Các hệ thống quản lý tri thức, Logic mô tả (Description Logics).

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter10_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 10.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_10.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch10_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 10.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter10_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
