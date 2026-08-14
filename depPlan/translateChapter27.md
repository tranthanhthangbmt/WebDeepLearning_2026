# Kế hoạch chi tiết dịch Chương 27: COMPUTER VISION (Thị giác máy tính)

## I. Mục tiêu
- Dịch nội dung **Chương 27: Computer Vision** từ sách tiếng Anh sang tiếng Việt.
- Bảo toàn cấu trúc HTML gốc, công thức toán học với MathJax (đặc biệt là các công thức về Xử lý ảnh, Mạng nơ-ron tích chập CNN, Hình học quang học), thẻ danh sách, in đậm/nghiêng.
- Tự động tách nhỏ nội dung thành nhiều file để dễ kiểm soát tiến độ, sau đó gộp lại.
- Dò tìm và chèn đúng vị trí hình ảnh (Figure 27.1, 27.2,...) vào bản dịch.
- Tích hợp kết quả vào repository hiện tại để sinh viên dễ dàng tra cứu qua web.

## II. Phân chia công việc (Divide and Conquer)
Tổng cộng chương 27 có 44 trang, được chia thành **9 phần (từ part 1 đến part 9)** để đảm bảo tính ổn định và chi tiết trong quá trình dịch thuật. Các phần này tương ứng với thư mục lưu trữ `TaiLieu/ebooks_Chapters_Vi3/Chapter_27_Computer Vision/`.

- **Phần 1 (`part1_27.html`):** 27.1 Introduction (Giới thiệu) và 27.2 Image Formation (27.2.1 The pinhole camera, 27.2.2 Lens systems, 27.2.3 Scaled orthographic projection).
- **Phần 2 (`part2_27.html`):** Tiếp tục 27.2 (27.2.4 Light and shading, 27.2.5 Color) và 27.3 Simple Image Features (27.3.1 Edges).
- **Phần 3 (`part3_27.html`):** Tiếp tục 27.3 (27.3.2 Texture, 27.3.3 Optical flow, 27.3.4 Segmentation of natural images) và mở đầu 27.4 Classifying Images.
- **Phần 4 (`part4_27.html`):** Tiếp tục 27.4 (27.4.1 Image classification with CNNs, 27.4.2 Why CNNs classify images well) và 27.5 Detecting Objects.
- **Phần 5 (`part5_27.html`):** 27.6 The 3D World (27.6.1 3D cues from multiple views, 27.6.2 Binocular stereopsis, 27.6.3 3D cues from a moving camera, 27.6.4 3D cues from one view).
- **Phần 6 (`part6_27.html`):** 27.7 Using Computer Vision (27.7.1 Understanding what people are doing, 27.7.2 Linking pictures and words).
- **Phần 7 (`part7_27.html`):** Tiếp tục 27.7 (27.7.3 Reconstruction from many views, 27.7.4 Geometry from a single view, 27.7.5 Making pictures).
- **Phần 8 (`part8_27.html`):** Tiếp tục 27.7 (27.7.6 Controlling movement with vision), Tóm tắt (Summary) và Ghi chú Lịch sử - Thư mục (Bibliographical and Historical Notes).
- **Phần 9 (`part9_27.html`):** Phần còn lại của Ghi chú Lịch sử - Thư mục (Bibliographical and Historical Notes) và Bài tập.

## III. Các bước thực hiện chi tiết

**Bước 1: Trích xuất và dịch từng phần (Translation phase)**
- Dùng kịch bản Python (`extract_pages_27.py`) để trích xuất văn bản từ tệp PDF thành các tệp `.txt` từ `part1_27.txt` đến `part9_27.txt`.
- Tiến hành dịch tuần tự: đọc từng file `partX_27.txt`, giữ nguyên ý nghĩa chuyên ngành và cấu trúc, xuất ra tệp `partX_27.html`.
- Áp dụng chặt chẽ định dạng thẻ HTML và MathJax cho công thức.
- Bắt giữ các chú thích hình ảnh định dạng `<p><strong>Hình 27.X</strong> ...</p>`.

**Bước 2: Gộp file (Merging)**
- Dùng hoặc tạo kịch bản Python `merge_27.py` để ghép tất cả 9 file `partX_27.html` theo thứ tự từ 1 đến 9.
- Lưu file tổng hợp vào `TaiLieu/ebooks_Chapters_Vi3/Chapter_27_Computer Vision/chapter_27_vi.html`.

**Bước 3: Chèn hình ảnh (Image Injection)**
- Chạy kịch bản chèn hình ảnh (`inject_figures_all_3.py` hoặc tạo file mới tương tự) để dò tìm các thẻ `<strong>Hình 27.X</strong>` trong file `chapter_27_vi.html`.
- Thay thế và chèn mã HTML hiển thị ảnh lấy từ thư mục `TaiLieu/Figures/Images/figure_27.X.jpg`.

**Bước 4: Tích hợp website (Integration)**
- Cập nhật tệp markdown `chapters/chapter_27_computer_vision.md`.
- Trỏ thẻ `<iframe>` trong mục `#### **Tiếng Việt**` đến đường dẫn chứa tệp `chapter_27_vi.html` (thêm tham số cache như `?v=1` nếu cần thiết).

**Bước 5: Kiểm tra (Verification)**
- Theo dõi bằng danh sách (Task List) và kiểm tra giao diện hiển thị web.
- Đảm bảo hình ảnh được căn giữa, văn bản dễ đọc và không có đoạn mã thừa.

---
**Một số lưu ý thuật ngữ:**
- Computer Vision -> Thị giác máy tính
- Image Formation -> Quá trình tạo ảnh
- Pinhole camera -> Máy ảnh lỗ kim
- Optical flow -> Luồng quang học
- Convolutional Neural Networks (CNNs) -> Mạng nơ-ron tích chập
- Stereopsis -> Thị giác âm nổi (hoặc Thị sai hai mắt)
- Image classification -> Phân loại hình ảnh
- Object detection -> Phát hiện đối tượng
