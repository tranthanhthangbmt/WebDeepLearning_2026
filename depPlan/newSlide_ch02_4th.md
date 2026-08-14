# Goal Description

Tiếp tục sự thành công của Chương 1, mục tiêu là tạo file trình chiếu LaTeX (`Chapter02_4th.tex`) cho Chương 2: **"Tác nhân thông minh" (Intelligent Agents)** dựa trên nội dung dịch tiếng Việt của phiên bản thứ 4 (4th Edition) AIMA.

Các slide sẽ kế thừa toàn bộ các tuỳ chỉnh đã sửa lỗi từ Chương 1 (dùng `\parbox` cho tiêu đề dài, thêm dòng trống ngăn cách các bullet point `\blob`, tự động tính toán kích cỡ ảnh) và phân bổ thời lượng hợp lý cho 90 phút giảng dạy đại học. Kế hoạch này cũng được lưu tại `depPlan/newSlide_ch02_4th.md`.

## User Review Required

> [!IMPORTANT]
> **Khối lượng nội dung Chương 2 khá dài (Đặc biệt là 5 mô hình Tác nhân).** Thầy hãy xem phần phân bổ thời lượng bên dưới, nếu cần cắt giảm hoặc ưu tiên mô hình nào (ví dụ: tập trung nhiều vào Utility-based và Learning Agent), vui lòng phản hồi.

> [!WARNING]
> **Sơ đồ các Tác nhân:** Chương 2 có rất nhiều sơ đồ minh họa quan trọng (ví dụ hình 2.9: Tác nhân phản xạ đơn giản, 2.11: Tác nhân phản xạ có mô hình, v.v.). Tôi sẽ trích xuất và hiển thị to rõ các `figure_2.x.jpg` (tương đương) từ thư mục `Figures/Images`. Thầy có muốn tôi chèn cả mã giả (Pseudocode) cho các tác nhân này vào slide không, hay chỉ cần sơ đồ cấu trúc khối là đủ?

## Proposed Changes

### 1. Cấu trúc và Phân bổ Bài giảng (90 phút)

Dự kiến khoảng 35 - 40 slide, bao gồm các phần:

1. **2.1 Tác nhân và Môi trường (10 phút - ~4 slide)**
   - Khái niệm Tác nhân (Agent), Cảm biến (Sensors), Cơ cấu chấp hành (Actuators).
   - Hàm tác nhân (Agent function) và Chương trình tác nhân (Agent program).
   - Ví dụ minh hoạ cơ bản: Tác nhân hút bụi (Vacuum-cleaner world).

2. **2.2 Hành vi tốt: Khái niệm hợp lý (Rationality) (15 phút - ~5 slide)**
   - Độ đo hiệu suất (Performance Measures).
   - Tính hợp lý (Định nghĩa Tác nhân hợp lý lý tưởng).
   - Sự toàn tri (Omniscience) vs. Tính hợp lý.
   - Học tập (Learning) và Tính tự chủ (Autonomy).

3. **2.3 Bản chất của Môi trường (20 phút - ~8 slide)**
   - Xác định môi trường tác vụ (Mô hình **PEAS**: Performance, Environment, Actuators, Sensors). Lấy ví dụ xe tự lái.
   - 7 thuộc tính của môi trường: 
     + Hoàn toàn/Một phần có thể quan sát (Fully / Partially observable)
     + Đơn/Đa tác nhân (Single / Multi-agent)
     + Tất định/Xác suất (Deterministic / Stochastic)
     + Tuần tự/Giai đoạn (Episodic / Sequential)
     + Tĩnh/Động (Static / Dynamic)
     + Rời rạc/Liên tục (Discrete / Continuous)
     + Đã biết/Chưa biết (Known / Unknown)

4. **2.4 Cấu trúc của Tác nhân (35 phút - ~15 slide)**
   - Khái quát chương trình tác nhân ($Agent = Architecture + Program$).
   - Phân tích 5 mô hình cơ bản (Sử dụng sơ đồ minh họa khối):
     1. Tác nhân phản xạ đơn giản (Simple reflex agents)
     2. Tác nhân phản xạ dựa trên mô hình (Model-based reflex agents)
     3. Tác nhân dựa trên mục tiêu (Goal-based agents)
     4. Tác nhân dựa trên tiện ích (Utility-based agents)
     5. Tác nhân học thuật (Learning agents - Critic, Learning element, Performance element, Problem generator).
   - So sánh cách hoạt động của các thành phần.

5. **Tổng kết & Q&A (10 phút - ~3 slide)**

### 2. Chi tiết các file sẽ thao tác

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\TaiLieu\slide_4th\Chapter02_4th.tex
- Tạo mã nguồn LaTeX cho toàn bộ bài giảng.
- Kế thừa lại macro `\heading` đã được sửa bằng `\parbox` để các tiêu đề dài tự động xuống dòng.
- Đảm bảo các item `\blob` kết thúc bằng `\\` để ngắt dòng đúng chuẩn.
- Dùng `\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]` cho các sơ đồ tác nhân để chiếm trọn vẹn khung hình trống của slide.

#### [NEW] D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Trí tuệ nhân tạo_UDA_2025\webTTNT_2026\depPlan\newSlide_ch02_4th.md
- Nội dung bản kế hoạch này sẽ được sao chép và lưu trữ làm tài liệu đối chiếu cho thầy.

## Verification Plan

### Automated Tests
- Kiểm tra tính hợp lệ của macro tự định nghĩa lại, tránh đụng độ (loại bỏ `amsmath`).

### Manual Verification
- Dịch và trích xuất PDF thông qua lệnh `pdflatex Chapter02_4th.tex`.
- Mở PDF kiểm tra trực quan sơ đồ các mô hình (ví dụ `figure_2.x.jpg`) đã được đưa vào kích thước lớn nhất có thể chưa.
- Kiểm tra không để các mục PEAS hoặc thuộc tính môi trường bị tràn trang.
