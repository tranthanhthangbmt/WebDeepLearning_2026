# Kế hoạch chi tiết dịch Chương 26: ROBOTICS (Robotics)

## I. Mục tiêu
- Dịch nội dung **Chương 26: Robotics** (Robot) từ sách tiếng Anh sang tiếng Việt.
- Bảo toàn cấu trúc HTML gốc, công thức toán học với MathJax (đặc biệt là các công thức SLAM, MDPs, PID controller), thẻ danh sách, in đậm/nghiêng.
- Tự động tách nhỏ nội dung thành nhiều file để dễ kiểm soát tiến độ, sau đó gộp lại.
- Dò tìm và chèn đúng vị trí hình ảnh (Figure 26.1, 26.2,...) vào bản dịch.
- Tích hợp kết quả vào repository hiện tại để sinh viên dễ dàng tra cứu qua web.

## II. Phân chia công việc (Divide and Conquer)
Tổng cộng chương 26 có 56 trang, được chia thành **11 phần (từ part 1 đến part 11)** để đảm bảo tính ổn định và chi tiết trong quá trình dịch thuật. Các phần này tương ứng với thư mục lưu trữ `TaiLieu/ebooks_Chapters_Vi3/Chapter_26_Robotics`.

- **Phần 1 (`part1_26.html`):** 26.1 Robots (Giới thiệu) và mở đầu 26.2 Robot Hardware.
- **Phần 2 (`part2_26.html`):** Tiếp tục phần cứng (Sensors, Producing motion) và 26.3 What kind of problem is robotics solving?
- **Phần 3 (`part3_26.html`):** 26.4 Robotic Perception (Localization, Mapping, SLAM).
- **Phần 4 (`part4_26.html`):** Mở đầu 26.5 Planning and Control (Configuration space, Motion planning).
- **Phần 5 (`part5_26.html`):** Tiếp tục 26.5 (Trajectory tracking, Optimal control).
- **Phần 6 (`part6_26.html`):** 26.6 Planning Uncertain Movements và mở đầu 26.7 Reinforcement Learning in Robotics.
- **Phần 7 (`part7_26.html`):** 26.8 Humans and Robots (Coordination, Predicting human action).
- **Phần 8 (`part8_26.html`):** Tiếp tục 26.8 (Inverse Reinforcement Learning).
- **Phần 9 (`part9_26.html`):** 26.9 Alternative Robotic Frameworks (Reactive controllers, Subsumption architectures).
- **Phần 10 (`part10_26.html`):** 26.10 Application Domains (Lĩnh vực ứng dụng).
- **Phần 11 (`part11_26.html`):** Tổng kết, Bài tập, Ghi chú lịch sử.

## III. Các bước thực hiện chi tiết

**Bước 1: Trích xuất và dịch từng phần (Translation phase)**
- File đầu vào là các file `.txt` từ `part1_26.txt` đến `part11_26.txt` đã được trích xuất tự động từ file PDF.
- Tiến hành dịch tuần tự: mỗi phần sẽ đọc file `partX_26.txt`, giữ nguyên ý nghĩa và cấu trúc, xuất ra file `partX_26.html`.
- Áp dụng các quy tắc về HTML markup và MathJax (dùng `$..$` hoặc `\(..\)` cho công thức trên dòng và `$$..$$` hoặc `\[..\]` cho công thức nhiều dòng).
- Bắt giữ các chú thích hình ảnh như `Hình 26.x (Figure 26.x)` dạng in đậm hoặc `<strong>`.

**Bước 2: Gộp file (Merging)**
- Tạo kịch bản Python `merge_26.py` để đọc và ghép tất cả 11 file `partX_26.html` theo thứ tự từ 1 đến 11.
- Lưu kết quả gộp vào file `TaiLieu/ebooks_Chapters_Vi3/Chapter_26_Robotics/chapter_26_vi.html`.

**Bước 3: Chèn hình ảnh (Image Injection)**
- Chạy kịch bản Python `inject_figures_all_3.py` để tự động dò tìm các thẻ `<strong>Hình 26.X</strong>` trong file `chapter_26_vi.html`.
- Kịch bản sẽ chèn mã HTML `<img src="TaiLieu/Figures/Images/figure_26.X.jpg" style="width: 100%;" ...>` vào dưới mỗi chú thích hình, thay thế chỗ trống hình ảnh.

**Bước 4: Tích hợp website (Integration)**
- Chỉnh sửa file markdown mô tả chương `chapters/chapter_26_robotics.md`.
- Thay đổi đường dẫn trong thẻ `<iframe>` dưới mục `#### **Tiếng Việt**` trỏ tới file `TaiLieu/ebooks_Chapters_Vi3/Chapter_26_Robotics/chapter_26_vi.html`.

**Bước 5: Cập nhật Task Log và Rà soát (Verification)**
- Sử dụng bảng theo dõi (Task List) để cập nhật tiến độ liên tục sau mỗi bước.
- Làm mới trình duyệt (hoặc báo cho giảng viên) để kiểm tra kết xuất cuối cùng trên web: văn bản có bị lặp/lỗi không, công thức toán có hiển thị đẹp không, hình ảnh đã lên chưa.

---
**Lưu ý khi dịch thuật ngữ Robotics:**
- Configuration space -> Không gian cấu hình
- Degrees of Freedom (DOF) -> Bậc tự do
- Forward/Inverse kinematics -> Động học thuận/nghịch
- Trajectory tracking -> Bám quỹ đạo
- Controller -> Bộ điều khiển
- Reinforcement Learning -> Học tăng cường
