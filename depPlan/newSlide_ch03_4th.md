# Goal Description

Tiếp nối chuỗi bài giảng, mục tiêu là tạo file trình chiếu LaTeX (`Chapter03_4th.tex`) cho **Chương 03: Giải quyết vấn đề bằng Tìm kiếm (Solving Problems by Searching)** dựa trên nội dung tiếng Việt của AIMA Phiên bản thứ 4. Các slide sẽ kế thừa tuỳ chỉnh `\parbox`, tự động xử lý hình ảnh và ngắt dòng hợp lý. File kế hoạch này được lưu tại `depPlan/newSlide_ch03_4th.md`.

## User Review Required

> [!WARNING]
> Chương 03 chứa các phần yêu cầu biểu diễn sơ đồ/đồ thị thuật toán rất nhiều. Tôi sẽ ưu tiên sử dụng `figure_03.x.jpg` (sơ đồ đồ thị) thay vì nhét chữ mô tả thuật toán. Thầy đồng ý với cách trình bày tối ưu trực quan này chứ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

1. **3.1 Các tác nhân giải quyết vấn đề (10 phút)**
   - Vấn đề là gì, Goal formulation.
2. **3.2 Các bài toán ví dụ (15 phút)**
   - Toy problems (8-puzzle, 8-queens) và Real-world problems (Định tuyến, Robot...).
3. **3.3 Thuật toán Tìm kiếm (15 phút)**
   - Cây tìm kiếm, đồ thị, trạng thái biên (frontier).
4. **3.4 Chiến lược Tìm kiếm Không thông tin (25 phút)**
   - Tìm kiếm theo chiều rộng (BFS), chiều sâu (DFS), sâu lặp dần (IDS), chi phí cực tiểu (UCS).
5. **3.5 Tìm kiếm Có thông tin (Heuristic) (20 phút)**
   - Tham lam (Greedy), Tìm kiếm A* (A* Search).
6. **3.6 Hàm Heuristic (5 phút)**
   - Đặc tính admissibility và consistency.

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter03_4th.tex
- Tạo mã nguồn LaTeX cho bài giảng Chương 03.
- Sử dụng cấu trúc `\heading` có parbox. Đảm bảo ảnh từ thư mục `Figures/Images` (ví dụ `figure_03.x.jpg`) hiển thị với `[width=0.95	extwidth, height=0.65	extheight, keepaspectratio]`.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan
ewSlide_ch03_4th.md
- File kế hoạch lưu trữ nội dung phân bổ bài giảng cho Chương 03.

## Verification Plan

### Manual Verification
- Chạy `pdflatex Chapter03_4th.tex` để sinh PDF.
- Đối chiếu PDF: Mọi đồ thị, biểu thức toán học và sơ đồ hiển thị tràn viền, rõ ràng.
