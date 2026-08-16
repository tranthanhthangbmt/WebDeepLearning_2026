# Kế hoạch Cập nhật Website: Nút Chuyển đổi Ngôn ngữ Thực hành (VN / EN)

Nhiệm vụ này yêu cầu cập nhật các trang markdown (`chuong_01.md` đến `chuong_19.md`) và file cấu hình chính (`index.html`) để thêm tính năng chuyển đổi ngôn ngữ linh hoạt cho các Jupyter Notebook thực hành.

## Yêu cầu
1. **Quét dữ liệu**: Tự động duyệt thư mục `TaiLieu/NotebookJupyter` để gom nhóm các file `.ipynb` theo từng chương.
2. **Sắp xếp**: Nếu một chương có nhiều file notebook, chúng sẽ được đánh số thứ tự (ví dụ: `01.1`, `01.2`, v.v.). Nếu không có số, chúng sẽ được sắp xếp mặc định theo tên.
3. **Fallback Logic (Dự phòng)**: Nếu phiên bản VN không tồn tại, tab VN sẽ hiển thị tạm bản EN của file đó (và ngược lại) để đảm bảo sinh viên luôn có tài liệu.
4. **Giao diện & Nút chuyển đổi (Toggle)**: Thêm nút chuyển đổi gắn cờ 🇻🇳 VN và 🇬🇧 EN ở góc phải của tab Thực hành. Mặc định luôn hiển thị bản VN.

## Đề xuất Thay đổi

### 1. Cập nhật `index.html`
Thêm hàm Javascript toàn cục `togglePracticeLang(lang)` vào cuối thẻ `<body>` để điều khiển việc ẩn/hiện danh sách notebook theo ngôn ngữ tương ứng.

#### [MODIFY] index.html
Sẽ chèn đoạn mã JS sau:
```javascript
window.togglePracticeLang = function(lang) {
  const vnList = document.getElementById('notebook-list-VN');
  const enList = document.getElementById('notebook-list-EN');
  const btnVn = document.getElementById('btn-vn');
  const btnEn = document.getElementById('btn-en');
  if(vnList) vnList.style.display = lang === 'VN' ? 'block' : 'none';
  if(enList) enList.style.display = lang === 'EN' ? 'block' : 'none';
  if(btnVn && btnEn) {
     btnVn.style.opacity = lang === 'VN' ? '1' : '0.5';
     btnEn.style.opacity = lang === 'EN' ? '1' : '0.5';
  }
}
```

### 2. Script Tự động Cập nhật Markdown (`update_web_practice.py`)
Tạo một script Python trong thư mục gốc. Script này sẽ:
- Lấy danh sách tất cả các file trong `TaiLieu/NotebookJupyter`.
- Phân tích cú pháp tên file bằng Regex để trích xuất `Chương`, `Số thứ tự` (nếu có), `Tên bài` và `Ngôn ngữ` (VN/EN).
- Duyệt qua từng file `chuong_XX.md` trong thư mục `docs/`.
- Xác định vị trí của khối `#### ** 💻 Thực hành **` và thay thế toàn bộ khối đó bằng mã HTML mới chứa nút Toggle và 2 danh sách (`<ul id="notebook-list-VN">` và `<ul id="notebook-list-EN">`).

### 3. Cập nhật hàng loạt các file `chuong_XX.md`
Chạy script Python để tự động cập nhật hàng loạt nội dung cho tất cả các chương. 
Khối HTML mới sẽ có dạng:
```html
#### ** 💻 Thực hành **
<div class="practice-container" style="...">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <h3>🚀 Bài tập Thực hành Jupyter Notebook</h3>
    <div class="lang-toggle">
      <button id="btn-vn" onclick="togglePracticeLang('VN')" style="opacity: 1;">🇻🇳 VN</button>
      <button id="btn-en" onclick="togglePracticeLang('EN')" style="opacity: 0.5;">🇬🇧 EN</button>
    </div>
  </div>
  ...
  <ul id="notebook-list-VN" style="display:block;"> ...danh sách VN... </ul>
  <ul id="notebook-list-EN" style="display:none;"> ...danh sách EN... </ul>
</div>
```

## User Review Required

> [!IMPORTANT]
> Script sẽ thay thế hoàn toàn nội dung từ dòng `#### ** 💻 Thực hành **` cho đến hết file (hoặc đến `<!-- tabs:end -->`) trong tất cả các file `chuong_XX.md`.
> Các liên kết trong nút "Mở trên Google Colab" sẽ được tạo tự động dựa trên cấu trúc đường dẫn repo Github của bạn (`tranthanhthangbmt/M-n-M-y-h-c_2026/blob/main/machineLearningWeb/TaiLieu/NotebookJupyter/...`).

Xin vui lòng xác nhận kế hoạch để tôi tiến hành viết script và cập nhật ngay.

tôi đã dịch các notebook jupyter sang tiếng Việt (VN). Trong tab Thực hành có nút bên phải trên, chọn mặc định là NV có lá cờ Việt Nam, người dùng có thể có thể click vào toggo này để nó chuyển sang EN (có lá cớ English nước Anh). Khi ở tiếng Việt nó hiển thị các notebook jupyter tiếng Việt, tiếng Anh thì tương tự, nếu không có file thì nó hiển thị tiếng Việt cũng được. Hãy kiểm tra và cập nhật vào web giúp tôi. Đồng thời có chương có nhiều notebook thì theo thứ tự XX.1 XX.2 XX.3