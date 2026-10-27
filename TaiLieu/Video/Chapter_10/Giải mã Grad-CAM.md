**Kỹ thuật 3: Biểu đồ nhiệt Grad-CAM (Class Activation Mapping dựa trên Gradient)** 1, 2 là kỹ thuật giải mã đỉnh cao giúp trả lời hai câu hỏi quan trọng trong Thị giác máy tính: **"Tại sao AI lại đưa ra kết quả phân loại này?"** và **"Vùng không gian nào trên bức ảnh thực tế đã quyết định điều đó?"** 3, 4\.

### 1\. Grad-CAM Kết hợp Kỹ thuật 1 & Kỹ thuật 2 Như Thế Nào?

Để hiểu tại sao Grad-CAM lại mạnh mẽ, hãy xem cách nó ghép nối ưu điểm của hai kỹ thuật trước đó:

* **Từ Kỹ thuật 1 (Feature Maps — Vị trí không gian 2D):** Kỹ thuật 1 cung cấp các bản đồ đặc trưng \\\\(A^k\\\\) ở **lớp tích chập cuối cùng** 2, 5\. Lớp này vẫn duy trì thông tin không gian 2D (chiều rộng \\\\(\\times\\\\) chiều cao) đồng thời chứa các khái niệm ngữ nghĩa bậc cao ở từng kênh \\\\(k\\\\) 2\. Tuy nhiên, Kỹ thuật 1 có nhược điểm: nó cho ta biết *ở đâu có đặc trưng gì*, nhưng không cho biết *đặc trưng đó có đóng góp vào kết quả phân loại nhãn mục tiêu hay không*.  
* **Từ Kỹ thuật 2 (Gradients — Độ nhạy/Tầm quan trọng của bộ lọc):** Kỹ thuật 2 sử dụng đạo hàm (gradient) để đo lường mức độ phản ứng 2, 6\. Grad-CAM tính đạo hàm của **điểm số lớp dự đoán** (ví dụ: lớp "Con voi") đối với từng kênh \\\\(k\\\\) của lớp tích chập cuối 2, 7\.  
* **Điểm giao nhau (Cốt lõi của Grad-CAM):**  
* Grad-CAM lấy từng bản đồ vị trí 2D của kênh \\\\(k\\\\) (từ Kỹ thuật 1\) nhân với **mức độ quan trọng** của kênh đó đối với nhãn dự đoán (thu được bằng cách lấy trung bình gradient như Kỹ thuật 2\) 2, 8\.

**Nói một cách ngắn gọn:**\\\\(\\text{Grad-CAM} \= \\text{Bản đồ vị trí (Kỹ thuật 1)} \\times \\text{Trọng số tầm quan trọng của kênh (Kỹ thuật 2)}\\\\) 2

### 2\. Quy trình 3 Bước Tạo Biểu đồ Nhiệt Grad-CAM

1. **Bước 1 — Trích xuất bản đồ đặc trưng cuối:**Đưa ảnh vào mạng và lấy bản đồ kích hoạt \\\\(A\\\\) ở lớp tích chập cuối cùng 2, 5\.  
2. **Bước 2 — Tính trọng số tầm quan trọng (\\\\(\\alpha\_k\\\\)):**Tính gradient của điểm số lớp mục tiêu đối với từng kênh \\\\(k\\\\) 2, 7\. Sau đó lấy trung bình toàn bộ không gian (Global Average Pooling) trên gradient đó để thu được con số \\\\(\\alpha\_k\\\\) đại diện cho mức độ đóng góp của kênh \\\\(k\\\\) 8, 9\.  
3. **Bước 3 — Nhân trọng số, Tổng hợp & Lọc qua ReLU:**Lấy tổng tất cả các kênh đã nhân trọng số \\\\(\\sum\_k \\alpha\_k A^k\\\\) và đưa qua hàm kích hoạt \\\\(\\text{ReLU}\\\\) 9:  
4. \\\\\\text{Heatmap} \= \\text{ReLU}\\left(\\sum\_k \\alpha\_k A^k\\right)\\\\  
5. *Hàm \\\\(\\text{ReLU}\\\\) giúp loại bỏ các giá trị âm, chỉ giữ lại các điểm ảnh có tác dụng **thúc đẩy tích cực** đến kết quả phân loại 9\.*

### 3\. Ví dụ Thực tế từ Giáo trình: Bức Ảnh Con Voi

Trong giáo trình (*Deep Learning with Python*), khi đưa bức ảnh gồm hai con voi (voi mẹ và voi con) trên đồng cỏ vào mạng Xception 2, 10:

1. Mô hình dự đoán nhãn **"African elephant" (Voi Châu Phi)** 11\.  
2. Khi áp dụng Grad-CAM và đè biểu đồ nhiệt lên ảnh gốc, các vùng màu nóng (đỏ, vàng) sáng rực lên tại **vùng tai, vòi và đầu của con voi** 3, 12\.  
3. **Phát hiện thú vị:** Biểu đồ nhiệt tập trung rất mạnh vào phần **tai của voi con** 4\. Điều này giải thích tại sao mạng phân loại đúng là *Voi Châu Phi* chứ không phải *Voi Châu Á* — vì Voi Châu Phi có đôi tai to đặc trưng, và mô hình đã thực sự "nhìn" vào đôi tai để đưa ra quyết định 4\!

### 4\. Ý nghĩa Thực tiễn của Grad-CAM

* **Phát hiện bẫy "Học vẹt" (Model Debugging):** Nếu mô hình phân loại ảnh "Con voi" nhưng Grad-CAM lại phát sáng ở vùng... thảm cỏ, kỹ sư sẽ biết ngay mô hình đang bị nhiễu do dữ liệu huấn luyện 13\.  
* **Độ tin cậy trong Y tế & Xe tự lái:** Giúp bác sĩ kiểm tra xem AI chẩn đoán bệnh dựa trên khối u thật hay dựa vào vết xước trên phim X-quang 14\.  
* **Định vị đối tượng không cần nhãn khung bao (Weakly Supervised Localization):** Giúp xác định vị trí vật thể trên ảnh mà không cần tốn công gán nhãn khung bao (bounding box) từ trước 1, 13\.

Hi vọng giải thích trên giúp bạn hình dung rõ nét cách Grad-CAM kết hợp cả vị trí không gian lẫn độ nhạy đạo hàm để "mở hộp đen" AI\!  
💡 Bạn có muốn tạo một slide bài giảng tóm tắt Chương 10 hay xem đoạn mã Keras 3 triển khai Grad-CAM chi tiết không?  
