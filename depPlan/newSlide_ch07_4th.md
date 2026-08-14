# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter07_4th.tex`) cho **Chương 07: Tác nhân Logic (Logical Agents)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch07_4th.md`.

## User Review Required

> [!WARNING]
> Chương 07 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_07.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **7.1 Tác nhân dựa trên Tri thức (10 phút)**
   - Cơ sở tri thức (Knowledge Base - KB), TELL và ASK.
2. **7.2 Thế giới Wumpus (Wumpus World) (15 phút)**
   - Mô tả môi trường: Vàng, Hố, Quái vật Wumpus.
3. **7.3 Logic học cơ bản (15 phút)**
   - Cú pháp (Syntax), Ngữ nghĩa (Semantics), Tính hệ quả (Entailment).
4. **7.4 Logic Mệnh đề (Propositional Logic) (15 phút)**
   - Các phép toán (AND, OR, NOT, IMPLIES, EQUIV).
5. **7.5 Chứng minh định lý Mệnh đề (20 phút)**
   - Suy diễn logic (Inference), Phân giải (Resolution), Forward/Backward chaining.
6. **7.6 - 7.7 Tác nhân Logic Mệnh đề (15 phút)**
   - SAT solvers và cách áp dụng vào Wumpus World.

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter07_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 07.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_07.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch07_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 07.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter07_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
