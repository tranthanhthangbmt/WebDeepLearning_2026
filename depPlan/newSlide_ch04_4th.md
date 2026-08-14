# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter04_4th.tex`) cho **Chương 04: Tìm kiếm trong Môi trường phức tạp (Search in Complex Environments)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch04_4th.md`.

## User Review Required

> [!WARNING]
> Chương 04 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_04.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **4.1 Tìm kiếm Cục bộ và Bài toán tối ưu hóa (30 phút)**
   - Leo đồi (Hill-climbing), Luyện kim nhân tạo (Simulated Annealing), Thuật toán Di truyền (Genetic Algorithms).
2. **4.2 Tìm kiếm Cục bộ trong Không gian Liên tục (15 phút)**
   - Gradient descent cơ bản.
3. **4.3 Tác nhân với Hành động Không tất định (15 phút)**
   - Tìm kiếm cây AND-OR (AND-OR search trees).
4. **4.4 Tìm kiếm với Quan sát Một phần (15 phút)**
   - Không gian niềm tin (Belief states).
5. **4.5 Môi trường chưa biết & Khám phá Trực tuyến (15 phút)**
   - Tác nhân trực tuyến (Online Search Agents).

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter04_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 04.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_04.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch04_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 04.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter04_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
