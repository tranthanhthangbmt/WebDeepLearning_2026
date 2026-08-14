# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter06_4th.tex`) cho **Chương 06: Tìm kiếm Đối kháng và Trò chơi (Adversarial Search and Games)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch06_4th.md`.

## User Review Required

> [!WARNING]
> Chương 06 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_06.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **6.1 Trò chơi và Lý thuyết Trò chơi (10 phút)**
   - Môi trường Multi-agent cạnh tranh, Zero-sum games.
2. **6.2 Quyết định tối ưu: Thuật toán Minimax (25 phút)**
   - Đánh giá giá trị node, Cây Minimax.
3. **6.3 Cắt tỉa Alpha-Beta (25 phút)**
   - Hiệu năng cắt tỉa và cách loại bỏ các nhánh không cần thiết.
4. **6.4 Quyết định thời gian thực không hoàn hảo (15 phút)**
   - Hàm đánh giá Heuristic, Giới hạn độ sâu (Depth limit).
5. **6.5 - 6.7 Trò chơi ngẫu nhiên & Khuyết thiếu thông tin (15 phút)**
   - Trò chơi có yếu tố xác suất (Expectiminimax).

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter06_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 06.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_06.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch06_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 06.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter06_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
