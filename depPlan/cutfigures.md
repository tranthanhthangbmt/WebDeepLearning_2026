# Kế hoạch Khắc phục Lỗi Hình ảnh Bị Trùng/Thiếu (Chương 11-23)

## Nguyên nhân gốc rễ
Sau khi phân tích tệp `global-figures.pdf`, tôi phát hiện ra rằng **rất nhiều trang trong PDF chứa nhiều hơn 1 hình ảnh**.
Ví dụ: Trang 81 chứa cả **Figure 11.1** (ở nửa trên) và **Figure 11.2** (ở nửa dưới).
Tuy nhiên, đoạn script trích xuất trước đó chỉ lấy hình ảnh đầu tiên nó tìm thấy trên trang (11.1) và lưu **toàn bộ trang đó** thành `figure_11.1.jpg`.
Hệ quả là:
1. Tại vị trí `Hình 11.1` trong web, nó hiển thị toàn bộ trang (gồm cả hình 11.1 và 11.2).
2. Tại vị trí `Hình 11.2` trong web, do không có file `figure_11.2.jpg` nào được tạo ra, nên nó bị bỏ trống!

## Đề xuất Giải pháp (Proposed Changes)

Tôi sẽ viết một Script thông minh hơn sử dụng thư viện `PyMuPDF` để tự động **cắt (crop)** các trang PDF dựa trên tọa độ của các dòng chú thích:
1. **Quét toàn bộ trang PDF**: Tìm kiếm tọa độ Y của tất cả các chữ `Figure X.Y`.
2. **Cắt ngang trang (Horizontal Slicing)**: 
   - Hình đầu tiên (VD 11.1): Cắt từ trên cùng của trang xuống đến dưới cùng của dòng chữ `Figure 11.1`.
   - Hình thứ hai (VD 11.2): Cắt từ dưới cùng của dòng chữ `Figure 11.1` xuống đến dưới cùng của dòng chữ `Figure 11.2`.
   - Lưu thành các file riêng biệt: `figure_11.1.jpg`, `figure_11.2.jpg`.
3. **Xử lý hình ảnh nằm ngang nhau (Side-by-side)**: Nếu có 2 hình nằm ngang nhau (tọa độ Y chênh lệch nhau < 50px), script sẽ giữ nguyên 2 hình trên cùng một khung ảnh ngang để không bị cắt phạm.

## Kế hoạch Thực thi (Verification Plan)
1. Chạy script để trích xuất và cắt lại toàn bộ hình ảnh cho các chương từ 11 đến 23. Các ảnh cũ sẽ bị ghi đè bằng ảnh đã cắt chính xác.
2. Chạy lại script chèn ảnh vào HTML. Lần này, do file `figure_11.2.jpg`, `11.4.jpg`... đã tồn tại, chúng sẽ tự động lấp đầy vào các khoảng trống trong web.
3. Kiểm tra lại kết quả hiển thị trên trình duyệt.

> [!IMPORTANT]
> Giải pháp này sẽ phân tách rõ ràng các hình ảnh bị dính chùm. Bạn có đồng ý để tôi tiến hành chạy Script xử lý này không?
