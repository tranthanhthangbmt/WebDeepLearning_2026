# 📝 BÀI KIỂM TRA GIỮA KỲ - MÔN HỌC SÂU (DEEP LEARNING)

> **Học phần:** Học Sâu (Deep Learning) — 3 Tín chỉ  
> **Đơn vị:** Khoa Công nghệ Thông tin & AI — Đại học Đông Á  
> **Giảng viên:** TS. Trần Thành Thắng  

---

### 📋 Thông Tin & Quy Chế Làm Bài

| Mục | Chi tiết |
| :--- | :--- |
| **Hình thức thi 1 (Trắc nghiệm)** | Trắc nghiệm trực tuyến **90 câu** (45 phút, bốc ngẫu nhiên cân bằng 12 chương, làm 1 lần duy nhất) |
| **Hình thức thi 2 (Thực hành)** | Lập trình trên **Jupyter Notebook / Google Colab** (45 phút, 2 Mã đề 01 và 02 phân bổ theo ca) |
| **Phạm vi kiến thức** | Bao quát toàn bộ **Chương 1 đến Chương 12** (Từ nền tảng Tensor đến Thị giác máy tính nâng cao) |
| **Mật khẩu làm bài** | Do Giảng viên cung cấp trực tiếp tại phòng thi (`Abcd1234` hoặc `UDA`) |
| **Giám sát & An ninh** | Khóa Copy/Paste, khóa chuột phải, F12; Bắt buộc kích hoạt JavaScript; Tự động ghi nhận số lần chuyển tab |
| **Đồng bộ cơ sở dữ liệu** | Điểm số tự động cập nhật trực tiếp lên Google Sheets: [Điểm Giữa kỳ AI Kế toán_15.10.2026](https://docs.google.com/spreadsheets/d/1jE5pqvoSDxC_NG6VmGULdFj6_Kj5miL1mInSCx4VrQc/edit?gid=0#gid=0) |

<div style="margin: 16px 0 24px 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
  <a href="quizzes/MidtermExam/index.html" target="_blank" class="btn-fullscreen" style="display: inline-flex; align-items: center; gap: 8px; background: #1a73e8; color: white !important; padding: 10px 18px; border-radius: 8px; font-weight: 600; text-decoration: none; box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3);">
    🚀 Mở Bài Thi Trắc Nghiệm Toàn Màn Hình
  </a>
  <a href="#/Chapters/kiem_tra_giua_ky?id=dethi-colab-canvas" style="display: inline-flex; align-items: center; gap: 8px; background: #7c3aed; color: white !important; padding: 10px 18px; border-radius: 8px; font-weight: 600; text-decoration: none; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);">
    📑 Xem 2 Đề Thi Thực Hành (Jupyter Notebook & Colab) ↓
  </a>
</div>

---

### 🖥️ Giao Diện Bài Thi Trắc Nghiệm Trực Tuyến

<iframe src="quizzes/MidtermExam/index.html" style="width: 100%; height: 92vh; min-height: 880px; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.08); background: #ffffff;" allowfullscreen></iframe>

---

<span id="dethi-colab-canvas"></span>

### 📑 Đề Thi Thực Hành Giữa Kỳ (Jupyter Notebook & Colab)

<div id="practical-exam-lock" style="background: #ffffff; border: 2px dashed #cbd5e1; border-radius: 14px; padding: 36px 20px; text-align: center; max-width: 640px; margin: 30px auto; box-shadow: 0 4px 20px rgba(0,0,0,0.06);">
  <div style="font-size: 3.5rem; margin-bottom: 12px; line-height: 1;">🔒</div>
  <h3 style="font-size: 1.45rem; font-weight: 800; color: #1e293b; margin-bottom: 10px;">KHU VỰC ĐỀ THI ĐÃ ĐƯỢC KHÓA BẢO MẬT</h3>
  <p style="color: #64748b; font-size: 0.95rem; max-width: 520px; margin: 0 auto 24px; line-height: 1.6;">
    Khu vực <strong>Đề thi thực hành giữa kỳ môn Học Sâu (Deep Learning)</strong> (Jupyter Notebook / Google Colab) được bảo mật. Sinh viên vui lòng nhập <strong>Mật khẩu làm bài</strong> do Giảng viên công bố tại phòng thi để mở khóa đề thi.
  </p>
  <div style="display: flex; gap: 10px; justify-content: center; max-width: 440px; margin: 0 auto; flex-wrap: wrap;">
    <div style="position: relative; flex: 1; min-width: 240px;">
      <input type="password" id="practical-exam-pass" placeholder="Nhập mật khẩu mở đề thi..." onkeydown="if(event.key==='Enter') window.checkPracticalExamPass();" oninput="window.clearPracticalExamErr();" style="width: 100%; box-sizing: border-box; padding: 12px 42px 12px 14px; border: 2px solid #cbd5e1; border-radius: 8px; font-size: 1rem; outline: none; transition: border-color 0.2s;">
      <button type="button" id="btn-toggle-exam-pwd" onclick="window.togglePracticalPassVisibility()" title="Ẩn/Hiện mật khẩu" style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); background: none; border: none; cursor: pointer; font-size: 1.15rem; color: #64748b; padding: 4px;">
        👁️
      </button>
    </div>
    <button type="button" id="btn-unlock-exam" onclick="window.checkPracticalExamPass()" style="background: #1a73e8; color: #ffffff; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 700; font-size: 0.95rem; cursor: pointer; box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3); transition: background 0.2s;">
      🔓 Mở Khóa Đề Thi
    </button>
  </div>
  <div id="practical-exam-err" style="display: none; color: #dc2626; font-size: 0.9rem; font-weight: 700; margin-top: 14px;">
    ❌ Mật khẩu không chính xác! Vui lòng kiểm tra lại.
  </div>
</div>

<div id="practical-exam-unlocked" style="display: none; margin-top: 24px;">

<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 12px 18px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;">
  <span style="color: #16a34a; font-weight: 700; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
    🔓 Đề thi thực hành đã được mở khóa thành công!
  </span>
  <button type="button" onclick="window.lockPracticalExam()" style="background: #ffffff; color: #64748b; border: 1px solid #cbd5e1; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; cursor: pointer;">
    🔒 Khóa lại
  </button>
</div>

<!-- tabs:start -->

#### **Mã Đề 01**

Dưới đây là bản thiết kế **Đề kiểm tra giữa kỳ thực hành trên Jupyter Notebook** (thời lượng 45 phút) dành cho lớp **IT23M**, bao quát trọn vẹn kiến thức cốt lõi từ **Chương 1 đến Chương 12**. 

Do thời gian làm bài gói gọn trong 45 phút, đề thi được thiết kế theo phương pháp **cung cấp mã khung (starter code / boilerplate)**: sinh viên không phải viết từ đầu các đoạn mã tải dữ liệu hay huấn luyện dài dòng, mà tập trung hoàn thiện các hàm/khối logic thuật toán cốt lõi để đảm bảo nộp và chấm điểm tự động nhanh chóng.

---

# ĐỀ KIỂM TRA GIỮA KỲ — HỌC PHẦN HỌC SÂU (DEEP LEARNING)
* **Lớp:** IT23M  
* **Thời gian làm bài:** 45 phút  
* **Hình thức:** Thực hành lập trình trên Jupyter Notebook / Google Colab (`.ipynb`)  
* **Phạm vi kiến thức:** Chương 1 đến Chương 12 (Từ toán học Tensor đến Thị giác máy tính nâng cao)  
* **Môi trường yêu cầu:** Python 3.10+, NumPy, TensorFlow / Keras 3

---

### HƯỚNG DẪN DÀNH CHO SINH VIÊN
1. Sinh viên mở file notebook `KiemTra_GiuaKy_IT23M_[MSSV].ipynb`.
2. Điền mã nguồn vào các khối có chú thích `# >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<`.
3. Tuyệt đối không thay đổi tên hàm, tên tham số và kích thước đầu ra đã được định nghĩa sẵn.
4. Chạy toàn bộ các ô kiểm thử (Test Cells) để đảm bảo không gặp lỗi cú pháp (`SyntaxError`) hoặc sai lệch số chiều (`Dimension Mismatch`).

---

## NỘI DUNG ĐỀ THI (THANG ĐIỂM 10)

### BÀI 1: Thao tác Tensor & Khởi tạo Tham số (3.0 Điểm — Mức độ: Dễ, ~10 phút)
*(Kiểm tra kiến thức Chương 1 – 3: Biểu diễn dữ liệu bằng Tensor, Tensor Slicing, Broadcasting và Khởi tạo trọng số)*

* **Yêu cầu 1.1 (1.5 điểm):** Cho một tensor ảnh đại diện cho một batch gồm 64 ảnh màu kích thước \\(128 \times 128 \times 3\\). Viết hàm `crop_and_batch_slice(batch_images)` trích xuất vùng trung tâm \\(64 \times 64\\) của 16 ảnh đầu tiên trong batch.
* **Yêu cầu 1.2 (1.5 điểm):** Viết hàm `manual_dense_forward(X, W, b)` thực hiện phép lan truyền tiến của một lớp Dense tuyến tính: \\(Z = X \cdot W + b\\) bằng NumPy/Keras Ops mà không dùng lớp có sẵn. Đảm bảo hàm xử lý đúng cơ chế **Broadcasting** khi cộng vector độ chệch (bias) \\(b\\) vào ma trận tích vô hướng.

```python
# --- STARTER CODE BÀI 1 ---
import numpy as np

def crop_and_batch_slice(batch_images):
    """
    Input: batch_images có shape (64, 128, 128, 3)
    Output: Tensor có shape (16, 64, 64, 3) trích xuất vùng trung tâm (từ hàng 32:96, cột 32:96) của 16 mẫu đầu
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass

def manual_dense_forward(X, W, b):
    """
    Input:
      - X: Tensor đầu vào có shape (batch_size, in_features)
      - W: Ma trận trọng số có shape (in_features, out_features)
      - b: Vector bias có shape (out_features,)
    Output:
      - Z: Tensor sau phép tính affine có shape (batch_size, out_features)
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass
```

---

### BÀI 2: Xây dựng Mô hình Phân lớp & Chống Overfitting (3.0 Điểm — Mức độ: Trung bình, ~15 phút)
*(Kiểm tra kiến thức Chương 4 – 7: Keras Functional/Sequential API, Callbacks, Loss/Activation cho phân loại và Regularization)*

Viết hàm `build_robust_classifier(input_shape, num_classes)` bằng **Keras Functional API** để giải quyết bài toán phân loại đa lớp:
1. **Kiến trúc mạng:**
   - Lớp đầu vào `Input(shape=input_shape)`.
   - Lớp ẩn thứ nhất `Dense` với 64 units, hàm kích hoạt `relu`.
   - Lớp **Dropout** với tỷ lệ \\(p = 0.3\\) để chống hiện tượng học vẹt (Overfitting).
   - Lớp ẩn thứ hai `Dense` với 32 units, hàm kích hoạt `relu`.
   - Lớp ngõ ra `Dense` phù hợp với số lượng `num_classes` và sử dụng hàm kích hoạt chuẩn mực cho phân loại đa lớp đơn nhãn (`softmax`).
2. **Biên dịch mô hình (Compile):**
   - Sử dụng bộ tối ưu `Adam` với learning rate \\(10^{-3}\\).
   - Hàm mất mát chuẩn mực `sparse_categorical_crossentropy` (hoặc `categorical_crossentropy`).
   - Độ đo đánh giá: `accuracy`.
3. **Cấu hình Callback:**
   - Tạo một `EarlyStopping` callback theo dõi `val_loss`, kiên nhẫn (`patience`) là 3 epoch, tự động phục hồi bộ trọng số tốt nhất (`restore_best_weights=True`).

```python
# --- STARTER CODE BÀI 2 ---
import keras
from keras import layers

def build_robust_classifier(input_shape=(100,), num_classes=10):
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    # 1. Khởi tạo các lớp theo Functional API
    # 2. Biên dịch (compile) mô hình
    # 3. Trả về: (model, early_stopping_callback)
    pass
```

---

### BÀI 3: Khối Tích Chập Hiện Đại — Residual & Separable Conv (2.5 Điểm — Mức độ: Khó, ~12 phút)
*(Kiểm tra kiến thức Chương 8 – 10: ConvNets, Kết nối thặng dư Residual Connection, Chuẩn hóa lô Batch Normalization và Trực quan hóa đặc trưng)*

Hiện thực một khối kiến trúc **Residual Block** cải tiến (lấy cảm hứng từ Mini-Xception) theo chuẩn mực Keras Functional API:
* Nhận đầu vào là tensor `x` có kích thước không gian \\((H, W, C)\\).
* Nhánh chính (Main path):
  1. Lớp tích chập tách rời chiều sâu `SeparableConv2D` với `filters=64`, `kernel_size=(3, 3)`, `padding="same"`.
  2. Lớp chuẩn hóa theo lô `BatchNormalization()`.
  3. Lớp kích hoạt `Activation("relu")`.
  4. Lớp `SeparableConv2D` thứ hai với `filters=64`, `kernel_size=(3, 3)`, `padding="same"`.
  5. Lớp `BatchNormalization()`.
* Nhánh tắt (Residual connection / Shortcut):
  - Nếu số kênh của `x` khác 64, chiếu `x` qua một lớp `Conv2D(64, (1, 1), padding="same")` để đồng nhất số chiều trước khi cộng.
* Đầu ra: Cộng nhánh chính với nhánh tắt bằng `layers.add([main_path, residual])`, sau đó đưa qua `Activation("relu")`.

```python
# --- STARTER CODE BÀI 3 ---
def build_residual_separable_block(input_tensor, num_filters=64):
    """
    Xây dựng một khối Residual Block kết hợp SeparableConv2D và BatchNormalization.
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass
```

---

### BÀI 4: Thị Giác Nâng Cao — Phân Đoạn Ảnh với Conv2DTranspose (1.5 Điểm — Mức độ: Vận dụng cao, ~8 phút)
*(Kiểm tra kiến thức Chương 11 – 12: Kiến trúc Encoder-Decoder, Phân giải tham số Conv2DTranspose và Bài toán phân đoạn ảnh 3 nhãn)*

Trong bài toán phân đoạn vật thể thú cưng (Oxford-IIIT Pet) ở Chương 11, mô hình cần bóc tách ảnh thành **3 nhãn pixel** (Vật thể, Phông nền, Đường viền biên).
* Viết hàm `build_segmentation_decoder(encoded_features)` đóng vai trò là **Decoder**:
  1. Nhận đầu vào là khối đặc trưng thu nhỏ `encoded_features` có kích thước \\((B, 25, 25, 128)\\).
  2. Sử dụng lớp tích chập chuyển vị `Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", activation="relu")` để phóng to kích thước không gian lên \\((B, 50, 50, 64)\\).
  3. Thêm một lớp `Conv2DTranspose(filters=32, kernel_size=3, strides=4, padding="same", activation="relu")` để phóng to lên kích thước gốc \\((B, 200, 200, 32)\\).
  4. Lớp tích chập cuối cùng `Conv2D(filters=3, kernel_size=3, padding="same", activation="softmax")` để dự đoán phân phối xác suất cho 3 nhãn trên từng pixel.

```python
# --- STARTER CODE BÀI 4 ---
def build_segmentation_decoder(encoded_features):
    """
    Input: Tensor đặc trưng nén shape (Batch, 25, 25, 128)
    Output: Tensor mặt nạ phân đoạn shape (Batch, 200, 200, 3) với hàm kích hoạt softmax
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass
```

---

## BAREM CHẤM ĐIỂM VÀ ĐÁP ÁN GỢI Ý (DÀNH CHO GIẢNG VIÊN)

| Bài | Tiêu chí đánh giá | Điểm |
| :--- | :--- | :---: |
| **Bài 1** | - Cắt lát đúng chiều không gian và chiều batch: `batch_images[:16, 32:96, 32:96, :]`.<br>- Tính đúng tích vô hướng `np.dot(X, W)` hoặc `X @ W` và cộng bias theo đúng broadcasting. | **3.0 đ**<br>*(1.5 đ)*<br>*(1.5 đ)* |
| **Bài 2** | - Cấu hình đúng Functional API: Input \\(\rightarrow\\) Dense(64) \\(\rightarrow\\) Dropout(0.3) \\(\rightarrow\\) Dense(32) \\(\rightarrow\\) Dense(num_classes, activation='softmax').<br>- Biên dịch đúng loss, optimizer, metrics.<br>- Thiết lập đúng Callback `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`. | **3.0 đ**<br>*(1.5 đ)*<br>*(0.8 đ)*<br>*(0.7 đ)* |
| **Bài 3** | - Sử dụng đúng `SeparableConv2D` kết hợp `BatchNormalization` ở 2 tầng của nhánh chính.<br>- Xử lý chuẩn nhánh tắt: Dùng `Conv2D(1x1)` khi lệch số kênh hoặc giữ nguyên `input_tensor`.<br>- Cộng hai nhánh bằng `layers.add` và kích hoạt `relu`. | **2.5 đ**<br>*(1.0 đ)*<br>*(1.0 đ)*<br>*(0.5 đ)* |
| **Bài 4** | - Cấu hình đúng các tham số `strides=2` và `strides=4` trong `Conv2DTranspose` để phục hồi đúng shape \\((200, 200)\\) từ \\((25, 25)\\).<br>- Lớp ngõ ra có `filters=3`, `padding="same"`, và kích hoạt `softmax`. | **1.5 đ**<br>*(1.0 đ)*<br>*(0.5 đ)* |

---

💡 Thầy/cô có muốn tôi chuyển đổi toàn bộ đề thi này thành file template Jupyter Notebook (`.ipynb`) kèm sẵn các đoạn mã test cases (assert assertions) để sinh viên chạy tự kiểm tra tại máy không?

#### **Mã Đề 02**

Dưới đây là **Đề kiểm tra giữa kỳ thực hành trên Jupyter Notebook — MÃ ĐỀ 02** dành cho lớp **IT23M**. Đề thi này có cấu trúc, độ khó và thời lượng (45 phút) hoàn toàn tương đương với Mã đề 01, nhưng thay đổi bài toán cụ thể và dạng dữ liệu từ **Chương 1 đến Chương 12** nhằm đảm bảo tính công bằng và chống sao chép khi cho thi song song hai đề.

---

# ĐỀ KIỂM TRA GIỮA KỲ — HỌC PHẦN HỌC SÂU (DEEP LEARNING)
* **Lớp:** IT23M  
* **MÃ ĐỀ:** **02**  
* **Thời gian làm bài:** 45 phút  
* **Hình thức:** Thực hành lập trình trên Jupyter Notebook / Google Colab (`.ipynb`)  
* **Phạm vi kiến thức:** Chương 1 đến Chương 12  
* **Môi trường yêu cầu:** Python 3.10+, NumPy, TensorFlow / Keras 3

---

### HƯỚNG DẪN DÀNH CHO SINH VIÊN
1. Sinh viên mở file notebook `KiemTra_GiuaKy_IT23M_De02_[MSSV].ipynb`.
2. Điền mã nguồn vào các khối có chú thích `# >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<`.
3. Tuyệt đối không thay đổi tên hàm, tên tham số và kiểu dữ liệu đầu ra được quy định sẵn.
4. Chạy toàn bộ các ô kiểm thử (Test Cells) để đảm bảo không gặp lỗi cú pháp (`SyntaxError`) hoặc lệch kích thước (`Dimension Mismatch`).

---

## NỘI DUNG ĐỀ THI (THANG ĐIỂM 10)

### BÀI 1: Xử lý Chuỗi Thời gian & Hàm Kích hoạt Tuyến tính (3.0 Điểm — Mức độ: Dễ, ~10 phút)
*(Kiểm tra kiến thức Chương 1 – 3: Tensor Rank-3 chuỗi thời gian, Tensor Slicing và Hàm kích hoạt ReLU thủ công)*

* **Yêu cầu 1.1 (1.5 điểm):** Một tensor chuỗi thời gian chứa dữ liệu cảm biến khí hậu có kích thước `(32, 100, 8)` (gồm 32 mẫu thử nghiệm, mỗi mẫu ghi nhận 100 bước thời gian, mỗi bước có 8 đặc trưng cảm biến). Viết hàm `slice_timeseries_window(data_tensor)` để trích xuất 24 bước thời gian cuối cùng của 4 cảm biến đầu tiên (từ cảm biến index 0 đến 3) cho tất cả các mẫu trong batch.
* **Yêu cầu 1.2 (1.5 điểm):** Viết hàm `manual_relu_activation(Z)` tính toán hàm kích hoạt phi tuyến **Rectified Linear Unit (ReLU)** mà không dùng hàm `keras.activations.relu` hay `tf.nn.relu`. Định nghĩa: \\(\text{ReLU}(z) = \max(0, z)\\). Đảm bảo xử lý được tensor có shape bất kỳ bằng NumPy hoặc Keras Ops.

```python
# --- STARTER CODE BÀI 1 ---
import numpy as np

def slice_timeseries_window(data_tensor):
    """
    Input: data_tensor có shape (32, 100, 8)
    Output: Tensor con có shape (32, 24, 4) chứa 24 bước thời gian cuối (từ bước 76 đến 99)
            và 4 cảm biến đầu tiên.
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass

def manual_relu_activation(Z):
    """
    Input: Z là một mảng NumPy/Tensor có shape bất kỳ (ví dụ: (batch_size, units))
    Output: Tensor kết quả sau khi áp dụng phép tính max(0, Z)
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass
```

---

### BÀI 2: Xây dựng Mô hình Hồi quy & Lưu Checkpoint Tự động (3.0 Điểm — Mức độ: Trung bình, ~15 phút)
*(Kiểm tra kiến thức Chương 4 – 7: Bài toán Hồi quy, Keras Functional API, Chuẩn hóa L2 Regularization, Callback ModelCheckpoint)*

Viết hàm `build_regression_pipeline(input_dim)` giải quyết bài toán dự đoán giá nhà (tương tự tập Boston/California Housing) bằng **Keras Functional API**:
1. **Kiến trúc mạng:**
   - Lớp đầu vào `Input(shape=(input_dim,))`.
   - Lớp ẩn thứ nhất `Dense` với 64 units, hàm kích hoạt `relu`, kèm theo **L2 Regularization** với trọng số phạt `l2(0.001)`.
   - Lớp ẩn thứ hai `Dense` với 32 units, hàm kích hoạt `relu`.
   - Lớp ngõ ra `Dense` phù hợp với bài toán **Hồi quy đơn biến** (Regression).
2. **Biên dịch mô hình (Compile):**
   - Sử dụng bộ tối ưu `RMSprop` với `learning_rate=0.001`.
   - Hàm mất mát chuẩn mực: Sai số toàn phương trung bình (`mean_squared_error` / `mse`).
   - Độ đo đánh giá: Sai số tuyệt đối trung bình (`mean_absolute_error` / `mae`).
3. **Cấu hình Callback:**
   - Tạo một `ModelCheckpoint` callback lưu mô hình vào đường dẫn `"best_regression_model.keras"`, chỉ lưu khi chỉ số `val_mae` đạt giá trị tốt nhất (`save_best_only=True`, `monitor="val_mae"`).

```python
# --- STARTER CODE BÀI 2 ---
import keras
from keras import layers, regularizers

def build_regression_pipeline(input_dim=13):
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    # 1. Khởi tạo mạng Functional API theo đúng cấu hình
    # 2. Biên dịch (compile) mô hình
    # 3. Tạo checkpoint callback
    # 4. Trả về: (model, checkpoint_callback)
    pass
```

---

### BÀI 3: Khối Tích Chập Downsampling & Residual Connection (2.5 Điểm — Mức độ: Khó, ~12 phút)
*(Kiểm tra kiến thức Chương 8 – 10: Downsampling bằng Stride vs Max-Pooling, Residual Projection, Batch Normalization)*

Hiện thực hàm `build_strided_residual_block(input_tensor, num_filters=128)` thực hiện giảm chiều không gian (downsampling) 2 lần bằng tích chập có bước nhảy:
* Nhận đầu vào là tensor `input_tensor` có shape \\((B, H, W, C_{in})\\).
* **Nhánh chính (Main path):**
  1. Lớp `Conv2D(num_filters, kernel_size=3, strides=2, padding="same")`.
  2. Lớp `BatchNormalization()`.
  3. Lớp kích hoạt `Activation("relu")`.
  4. Lớp `Conv2D(num_filters, kernel_size=3, strides=1, padding="same")`.
  5. Lớp `BatchNormalization()`.
* **Nhánh tắt (Residual connection / Shortcut):**
  - Do nhánh chính đã thu nhỏ kích thước không gian đi một nửa (\\(H/2, W/2\\)) và đổi số kênh thành `num_filters`, hãy chiếu `input_tensor` qua lớp `Conv2D(num_filters, kernel_size=1, strides=2, padding="same")` để kích thước hai nhánh khớp hoàn toàn.
* **Đầu ra:** Cộng hai nhánh bằng `layers.add([main_path, residual])`, kết thúc bằng `Activation("relu")`.

```python
# --- STARTER CODE BÀI 3 ---
def build_strided_residual_block(input_tensor, num_filters=128):
    """
    Xây dựng khối Residual Block vừa tăng số kênh vừa thu nhỏ chiều không gian (strides=2).
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass
```

---

### BÀI 4: Thị Giác Nâng Cao — Xây dựng Bounding Box Output (1.5 Điểm — Mức độ: Vận dụng cao, ~8 phút)
*(Kiểm tra kiến thức Chương 11 – 12: Kiến trúc mạng Đa đầu ra / Multi-output Network cho bài toán Phát hiện Đối tượng - Object Detection)*

Trong bài toán phát hiện đối tượng cơ bản (Chương 12), mạng nơ-ron nhận một khối đặc trưng nén `feature_maps` kích thước \\((B, 7, 7, 512)\\) và cần đồng thời dự đoán:
1. **Nhãn lớp của vật thể** (Phân loại).
2. **Tọa độ hộp bao** (Bounding box regression: `[ymin, xmin, ymax, xmax]`).

Viết hàm `build_detection_head(feature_maps, num_classes=10)`:
1. Dùng `GlobalAveragePooling2D()` để làm phẳng khối đặc trưng \\(7 \times 7 \times 512\\) thành vector 512 chiều.
2. Thêm một lớp `Dense(128, activation="relu")` làm tầng biểu diễn chung.
3. Tạo **Đầu ra 1 (`class_output`)**: Lớp `Dense(num_classes, activation="softmax", name="class_prediction")`.
4. Tạo **Đầu ra 2 (`box_output`)**: Lớp `Dense(4, activation="sigmoid", name="box_prediction")` (tọa độ chuẩn hóa trong khoảng \\(\\)).
5. Trả về cặp `[class_output, box_output]`.

```python
# --- STARTER CODE BÀI 4 ---
def build_detection_head(feature_maps, num_classes=10):
    """
    Input: feature_maps có shape (Batch, 7, 7, 512)
    Output: Danh sách chứa 2 tensor ngõ ra [class_output, box_output]
    """
    # >>> VIẾT CODE CỦA BẠN VÀO ĐÂY <<<
    pass
```

---

## BAREM CHẤM ĐIỂM VÀ ĐÁP ÁN GỢI Ý — MÃ ĐỀ 02

| Bài | Tiêu chí đánh giá | Điểm |
| :--- | :--- | :---: |
| **Bài 1** | - Cắt lát đúng chiều time và feature: `data_tensor[:, -24:, :4]` hoặc `data_tensor[:, 76:100, 0:4]`.<br>- Tính đúng hàm ReLU: `np.maximum(0, Z)` hoặc `np.where(Z > 0, Z, 0)`. | **3.0 đ**<br>*(1.5 đ)*<br>*(1.5 đ)* |
| **Bài 2** | - Cấu hình Functional API đúng bài toán hồi quy (Lớp cuối `Dense(1)` không kích hoạt hoặc `activation=None`).<br>- Thêm đúng `kernel_regularizer=regularizers.l2(0.001)`.<br>- Biên dịch đúng `loss='mse'` và `metrics=['mae']`.<br>- Cấu hình đúng `ModelCheckpoint` theo dõi `val_mae` và `save_best_only=True`. | **3.0 đ**<br>*(1.2 đ)*<br>*(0.5 đ)*<br>*(0.5 đ)*<br>*(0.8 đ)* |
| **Bài 3** | - Nhánh chính: Cấu hình đúng `Conv2D(strides=2)` ở tầng 1, `Conv2D(strides=1)` ở tầng 2, kèm BatchNorm.<br>- Nhánh tắt: Sử dụng đúng `Conv2D(128, 1, strides=2, padding="same")` để giảm chiều không gian khớp với nhánh chính.<br>- Cộng `layers.add` và bọc qua `Activation("relu")`. | **2.5 đ**<br>*(1.0 đ)*<br>*(1.0 đ)*<br>*(0.5 đ)* |
| **Bài 4** | - Dùng đúng `GlobalAveragePooling2D()` biến đổi tensor 3D thành 1D.<br>- Đầu ra phân loại dùng `Dense(num_classes, activation='softmax')`.<br>- Đầu ra tọa độ hộp bao gồm 4 giá trị và dùng `activation='sigmoid'` để chuẩn hóa về \\(\\). | **1.5 đ**<br>*(0.5 đ)*<br>*(0.5 đ)*<br>*(0.5 đ)* |

---

📦 Thầy/cô có muốn tôi xuất luôn đoạn mã script Python để tự động tạo 2 file Jupyter Notebook (`De_01.ipynb` và `De_02.ipynb`) kèm sẵn các câu lệnh `assert` kiểm thử đầu vào/đầu ra không?

<!-- tabs:end -->

</div>

<script>
(function() {
    const PRACTICAL_PASSWORD = "Abcd1234";

    window.togglePracticalPassVisibility = function() {
        const inp = document.getElementById('practical-exam-pass');
        if (inp) {
            inp.type = (inp.type === 'password') ? 'text' : 'password';
        }
    };

    window.clearPracticalExamErr = function() {
        const err = document.getElementById('practical-exam-err');
        const inp = document.getElementById('practical-exam-pass');
        if (err) err.style.display = 'none';
        if (inp) inp.style.borderColor = '#cbd5e1';
    };

    window.checkPracticalExamPass = function() {
        const inp = document.getElementById('practical-exam-pass');
        const err = document.getElementById('practical-exam-err');
        if (!inp) return;
        const val = inp.value.trim();
        if (val === PRACTICAL_PASSWORD || val === PRACTICAL_PASSWORD.toLowerCase() || val.toUpperCase() === "UDA") {
            sessionStorage.setItem('dl_practical_exam_unlocked', 'true');
            window.applyPracticalExamState();
        } else {
            if (err) err.style.display = 'block';
            inp.style.borderColor = '#dc2626';
            inp.focus();
        }
    };

    window.lockPracticalExam = function() {
        sessionStorage.removeItem('dl_practical_exam_unlocked');
        window.applyPracticalExamState();
    };

    window.applyPracticalExamState = function() {
        const isUnlocked = (sessionStorage.getItem('dl_practical_exam_unlocked') === 'true');
        const lockBox = document.getElementById('practical-exam-lock');
        const contentBox = document.getElementById('practical-exam-unlocked');
        if (lockBox && contentBox) {
            if (isUnlocked) {
                lockBox.style.display = 'none';
                contentBox.style.display = 'block';
                setTimeout(function() {
                    window.dispatchEvent(new Event('resize'));
                }, 100);
            } else {
                lockBox.style.display = 'block';
                contentBox.style.display = 'none';
                const inp = document.getElementById('practical-exam-pass');
                if (inp) inp.value = '';
                const err = document.getElementById('practical-exam-err');
                if (err) err.style.display = 'none';
            }
        }
    };

    // Tự động kiểm tra trạng thái ngay khi trang nạp xong
    setTimeout(window.applyPracticalExamState, 50);
})();
</script>

---

*Lưu ý: Sinh viên cần tuân thủ nghiêm túc quy chế kiểm tra. Mọi bài thi có dấu hiệu sao chép hoặc can thiệp kỹ thuật sẽ bị xử lý theo quy định học vụ của Nhà trường.*
