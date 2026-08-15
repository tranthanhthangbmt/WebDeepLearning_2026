const questions = [
  {
    type: "mcq",
    question: "Phát hiện đối tượng (Object Detection) khác với Phân loại hình ảnh (Image Classification) ở điểm cốt lõi nào?",
    options: [
      "Phát hiện đối tượng chỉ sử dụng hình ảnh trắng đen, phân loại dùng ảnh màu.",
      "Phát hiện đối tượng không gán nhãn lớp mà chỉ vẽ các đường viền xung quanh.",
      "Phát hiện đối tượng cho biết tên đối tượng VÀ vị trí của chúng thông qua hộp giới hạn.",
      "Phát hiện đối tượng chỉ áp dụng cho video, phân loại áp dụng cho ảnh tĩnh."
    ],
    correctAnswer: 2,
    explanation: "Phân loại hình ảnh chỉ cho biết 'có gì' trong ảnh. Phát hiện đối tượng giải quyết thêm câu hỏi 'ở đâu' bằng cách vẽ bounding boxes (hộp giới hạn) và gán nhãn cho từng hộp.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Tác vụ nào sau đây KHÔNG phải là một ứng dụng phổ biến của phát hiện đối tượng (Object Detection)?",
    options: [
      "Đếm số lượng (Counting) các cá thể có trong một bức ảnh.",
      "Theo dõi (Tracking) sự di chuyển của các vật thể trong một đoạn video.",
      "Cắt xén (Cropping) vùng ảnh chứa đối tượng để gửi cho mô hình nhận dạng chữ (OCR).",
      "Tái tạo âm thanh không gian (Spatial audio generation) dựa trên hình ảnh vật lý."
    ],
    correctAnswer: 3,
    explanation: "Ba ứng dụng chính yếu của phát hiện đối tượng là Counting, Tracking và Cropping. Tái tạo âm thanh thuộc lĩnh vực xử lý tín hiệu âm thanh, không liên quan đến bài toán này.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "So với Phân đoạn hình ảnh (Image Segmentation), ưu điểm lớn nhất của Phát hiện đối tượng (Object Detection) là gì?",
    options: [
      "Độ chính xác ở mức độ pixel (điểm ảnh) cao hơn rất nhiều.",
      "Khả năng chạy nhanh hơn nhiều và tiết kiệm chi phí gán nhãn dữ liệu.",
      "Có khả năng nhận diện các vật thể bị che khuất một phần (occlusion).",
      "Tự động tạo ra các mô hình 3D cho các đối tượng được nhận diện."
    ],
    correctAnswer: 1,
    explanation: "Mặc dù Phân đoạn (Segmentation) cung cấp nhiều thông tin hơn (ở cấp độ pixel), Phát hiện đối tượng (Detection) lại chạy nhanh hơn đáng kể và việc vẽ hộp giới hạn (bounding boxes) để gán nhãn tốn ít công sức hơn so với tô màu từng pixel.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Mô hình R-CNN (Region-based CNN) thuộc nhóm kiến trúc phát hiện đối tượng nào?",
    options: [
      "Máy dò một giai đoạn (Single-stage detectors).",
      "Máy dò hai giai đoạn (Two-stage detectors).",
      "Máy dò siêu phân giải (Super-resolution detectors).",
      "Máy dò theo cơ chế tự chú ý (Self-attention detectors)."
    ],
    correctAnswer: 1,
    explanation: "R-CNN là nền tảng của các mô hình phát hiện đối tượng hai giai đoạn (Two-stage). Giai đoạn 1: Đề xuất vùng (Region proposals). Giai đoạn 2: Phân loại từng vùng.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong giai đoạn đầu của máy dò R-CNN, thuật ngữ 'Region proposals' (Đề xuất khu vực) nghĩa là gì?",
    options: [
      "Một danh sách các từ khóa mô tả ngữ cảnh của toàn bộ bức ảnh đầu vào.",
      "Các hộp giới hạn (boxes) bao quanh những khu vực trông có vẻ giống một vật thể.",
      "Một mạng nơ-ron nhỏ dùng để tính toán tốc độ khung hình (FPS) cho video.",
      "Các điểm ảnh ngẫu nhiên được chọn để làm tâm cho việc tính toán tích chập."
    ],
    correctAnswer: 1,
    explanation: "Trong R-CNN, mô hình trước tiên tạo ra hàng ngàn 'đề xuất khu vực' — tức là các hộp giới hạn (boxes) bao quanh những vùng ảnh mà thuật toán (như Selective Search) cho là có chứa một vật thể nào đó.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Tại sao phương pháp phát hiện hai giai đoạn (như R-CNN) lại KHÔNG phù hợp cho các ứng dụng thời gian thực (real-time)?",
    options: [
      "Vì nó không thể phân loại được hình ảnh màu, chỉ hoạt động trên ảnh đen trắng.",
      "Vì độ chính xác của nó quá thấp, dẫn đến việc phải tính toán lại liên tục.",
      "Vì nó đòi hỏi phân loại hàng ngàn bản vá (patches) cho mỗi một bức ảnh duy nhất.",
      "Vì thuật toán này luôn yêu cầu kết nối Internet băng thông cao để hoạt động."
    ],
    correctAnswer: 2,
    explanation: "Sự cồng kềnh của R-CNN nằm ở việc nó tạo ra hàng ngàn đề xuất (proposals) và phải chạy một mạng CNN phân loại qua *từng* đề xuất đó cho mỗi bức ảnh. Điều này làm cho tốc độ xử lý rất chậm.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Dòng mô hình YOLO (You Only Look Once) thuộc loại kiến trúc phát hiện đối tượng nào?",
    options: [
      "Máy dò hai giai đoạn (Two-stage detectors).",
      "Máy dò ba giai đoạn (Three-stage detectors).",
      "Máy dò dựa trên đồ thị (Graph-based detectors).",
      "Máy dò một giai đoạn (Single-stage detectors)."
    ],
    correctAnswer: 3,
    explanation: "YOLO (và SSD, RetinaNet) là các máy dò một giai đoạn (Single-stage detectors). Nó dự đoán trực tiếp cả hộp giới hạn lẫn nhãn lớp trong cùng một lần quét qua mạng CNN.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Tập dữ liệu COCO (Common Objects in Context) thường được sử dụng cho mục đích gì?",
    options: [
      "Huấn luyện các mô hình tạo sinh ngôn ngữ tự nhiên như ChatGPT.",
      "Đóng vai trò làm bài kiểm tra hiệu năng chuẩn cho các thuật toán âm thanh.",
      "Làm tập dữ liệu chuẩn mực để huấn luyện và đánh giá các mô hình phát hiện đối tượng.",
      "Cung cấp hàng tỷ hình ảnh y tế (X-quang, MRI) cho các bệnh viện nghiên cứu."
    ],
    correctAnswer: 2,
    explanation: "COCO là một trong những bộ dữ liệu chuẩn và phổ biến nhất thế giới dành cho các bài toán Computer Vision, đặc biệt là Phát hiện đối tượng (Object Detection) và Phân đoạn hình ảnh (Segmentation).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong tập dữ liệu COCO được đề cập, có tất cả bao nhiêu nhãn (labels) đối tượng có thể có?",
    options: [
      "Chỉ có 2 nhãn (Có đối tượng hoặc Không có đối tượng).",
      "Khoảng 10 nhãn tương ứng với 10 loại động vật phổ biến.",
      "Tổng cộng 91 nhãn do những người tạo ra tập dữ liệu COCO lựa chọn.",
      "Hơn 10.000 nhãn tương ứng với mọi danh từ trong từ điển tiếng Anh."
    ],
    correctAnswer: 2,
    explanation: "COCO dataset cung cấp tổng cộng 91 lớp (danh mục) đối tượng phổ biến trong ngữ cảnh thực tế (như xe hơi, người, chó, mèo, cốc, v.v.).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong mô hình YOLO, cơ chế chia lưới (grid) hoạt động như thế nào?",
    options: [
      "Chia bức ảnh thành nhiều ô vuông nhỏ, mỗi ô chịu trách nhiệm phát hiện vật thể có tâm rơi vào ô đó.",
      "Chia bức ảnh thành 4 góc phần tư và chạy song song 4 mạng nơ-ron khác nhau để phân tích độc lập.",
      "Áp dụng một lưới điểm ảnh ngẫu nhiên để tăng cường dữ liệu trước khi đưa vào mô hình máy học.",
      "Chia ảnh thành các lớp màu (Red, Green, Blue) riêng biệt và hợp nhất chúng ở bước cuối cùng."
    ],
    correctAnswer: 0,
    explanation: "YOLO hoạt động bằng cách chia hình ảnh đầu vào thành một lưới (grid). Mỗi ô lưới (grid cell) sẽ chịu trách nhiệm dự đoán hộp giới hạn nếu *tâm* của đối tượng rơi vào bên trong ô lưới đó.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Ngoài các thông số tọa độ của hộp giới hạn (x, y, w, h), YOLO còn dự đoán thêm một giá trị nào đi kèm với mỗi hộp?",
    options: [
      "Mã màu Hex (màu sắc) của vật thể bên trong hộp.",
      "Trọng lượng ước tính (đơn vị kg) của vật thể đó.",
      "Điểm số tin cậy (Confidence score) về sự tồn tại của vật thể.",
      "Góc quay (Rotation angle) của vật thể so với trục hoành."
    ],
    correctAnswer: 2,
    explanation: "Cùng với thông số vị trí (x, y, w, h), mô hình YOLO dự đoán một 'điểm tin cậy' (Confidence score). Điểm này cho biết khả năng có vật thể thực sự ở đó và mức độ chính xác của hộp giới hạn so với thực tế (IoU).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Khi sử dụng ResNet làm xương sống (backbone) cho YOLO, tại sao tác giả lại nhấn mạnh việc sử dụng lớp tích chập có sải bước (strided convolutions) thay vì pooling?",
    options: [
      "Strided convolutions giúp mô hình chạy mượt mà hơn trên các vi xử lý cấp thấp.",
      "Pooling chỉ phù hợp cho việc phân tích âm thanh, không áp dụng cho hình ảnh.",
      "Strided convolutions bảo toàn thông tin về vị trí không gian tốt hơn cho việc phát hiện hộp giới hạn.",
      "Pooling yêu cầu bộ nhớ RAM gấp đôi so với Strided convolutions trong quá trình huấn luyện."
    ],
    correctAnswer: 2,
    explanation: "Giống như bài toán phân đoạn, bài toán phát hiện đối tượng cần giữ lại thông tin về vị trí (không gian). Pooling làm mất đi thông tin cục bộ, trong khi strided convolutions nén dữ liệu nhưng vẫn duy trì nhận thức về vị trí tương đối.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Đầu ra của mô hình YOLO được thiết kế có dạng Grid (ví dụ 6x6) nhân với `(num_labels + 5)`. Con số '5' ở đây đại diện cho gì?",
    options: [
      "5 góc độ quan sát khác nhau (trên, dưới, trái, phải, trực diện) của bức ảnh đầu vào.",
      "4 thông số tọa độ hộp giới hạn (x, y, w, h) và 1 điểm số tin cậy (confidence).",
      "5 cấp độ sáng khác nhau được mô hình áp dụng để bù trừ cho điều kiện thiếu sáng.",
      "5 bộ lọc (filters) mặc định được dùng cho các lớp Conv2DTranspose cuối cùng."
    ],
    correctAnswer: 1,
    explanation: "Mỗi ô lưới xuất ra các xác suất lớp (`num_labels` = 91) cộng thêm 5 giá trị cho hộp dự đoán: `x, y` (tâm), `w, h` (kích thước), và `confidence` (độ tin cậy).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong hàm Loss (tổn thất) của YOLO, tác giả làm gì để ngăn các ô lưới rỗng (không chứa vật thể) áp đảo quá trình huấn luyện?",
    options: [
      "Xóa bỏ hoàn toàn các ô lưới rỗng khỏi bộ nhớ GPU trước khi tính toán đạo hàm.",
      "Nhân đôi hàm loss của các ô lưới chứa vật thể lên gấp 10 lần so với mức bình thường.",
      "Chia tỷ lệ giảm (scale down) tổn thất độ tin cậy của các ô rỗng đi một nửa (hệ số 0.5).",
      "Thay thế các ô rỗng bằng những vật thể ngẫu nhiên được lấy từ thư viện ảnh mạng."
    ],
    correctAnswer: 2,
    explanation: "Vì phần lớn hình ảnh là cảnh nền (background), số ô rỗng rất nhiều. Để ngăn loss của việc dự đoán 'không có gì' lấn át tín hiệu học của các vật thể thực, tác giả áp dụng hệ số scale = 0.5 cho phần confidence loss tại các ô rỗng.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Để đo lường độ chính xác của hộp giới hạn dự đoán so với hộp thực tế (Ground truth) trong hàm Loss, YOLO sử dụng chỉ số nào làm mục tiêu (target) cho độ tin cậy?",
    options: [
      "Mean Absolute Error (MAE).",
      "Mean Squared Error (MSE).",
      "Intersection over Union (IoU).",
      "Peak Signal-to-Noise Ratio (PSNR)."
    ],
    correctAnswer: 2,
    explanation: "Nếu ô lưới chứa vật thể, thay vì chỉ gán mục tiêu độ tin cậy (target confidence) là 1.0, YOLO gán nó bằng giá trị IoU giữa hộp dự đoán và hộp thực tế. Điều này khuyến khích mô hình học cách tinh chỉnh hộp sao cho khít nhất (IoU cao nhất).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong tính toán Box Loss của YOLO, tại sao tác giả lại lấy căn bậc hai (square root) của chiều rộng (w) và chiều cao (h) trước khi trừ?",
    options: [
      "Để giảm thiểu số lượng tham số cần thiết trong mạng neural mạng kết nối đầy đủ (Dense layer).",
      "Để ngăn chặn sai số ở các hộp giới hạn lớn áp đảo quá mức sai số ở các hộp giới hạn nhỏ.",
      "Để tương thích với các thuật toán tối ưu hóa (Optimizers) như Adam hoặc RMSprop.",
      "Để đáp ứng yêu cầu toán học của hàm kích hoạt Softmax ở bước xuất ra dữ liệu cuối."
    ],
    correctAnswer: 1,
    explanation: "Lấy căn bậc hai giúp nén sự khác biệt ở các giá trị lớn. Lệch 10 pixel ở một hộp lớn (500x500) không nghiêm trọng bằng lệch 10 pixel ở một hộp nhỏ (20x20). Căn bậc hai giúp làm giảm khoảng cách này.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Lý do chính mà việc xử lý dữ liệu (Dataset) hình ảnh nên dùng thư viện `tf.data` thay vì load toàn bộ vào RAM (bộ nhớ) là gì?",
    options: [
      "Vì RAM thường có dung lượng nhỏ hơn nhiều so với tổng kích thước giải nén của hàng vạn bức ảnh, dẫn đến tràn bộ nhớ (Out of Memory).",
      "Vì thư viện `tf.data` có khả năng tự động sửa chữa các hình ảnh bị hỏng hoặc mất nét do lỗi camera.",
      "Vì `tf.data` cung cấp giao diện đồ họa đẹp mắt giúp người dùng theo dõi tiến trình load ảnh một cách dễ dàng.",
      "Vì sử dụng `tf.data` giúp mô hình tự động chuyển từ bài toán phát hiện đối tượng sang bài toán nhận dạng giọng nói."
    ],
    correctAnswer: 0,
    explanation: "Tập COCO dung lượng 18GB ảnh nén (JPEG). Khi giải nén thành tensor (NumPy array/Float32), kích thước sẽ phình to gấp nhiều lần. Tải hết vào RAM một lúc sẽ gây tràn RAM (OOM). `tf.data` giúp load từng batch nhỏ (streaming) rất hiệu quả.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Phương thức `.prefetch(2)` trong cấu trúc đường ống dữ liệu `tf.data` có tác dụng gì?",
    options: [
      "Nhân đôi số lượng hình ảnh huấn luyện thông qua kỹ thuật tăng cường dữ liệu (Data augmentation).",
      "Tự động gửi thông báo lỗi nếu có bất kỳ 2 hình ảnh nào trong thư mục bị thiếu nhãn.",
      "Cho phép CPU tải và chuẩn bị trước 2 batch dữ liệu trong lúc GPU đang xử lý batch hiện tại.",
      "Giới hạn mô hình chỉ chạy qua tập dữ liệu chính xác 2 lần (2 epochs) để tránh overfitting."
    ],
    correctAnswer: 2,
    explanation: "`prefetch(2)` giúp hệ thống chạy bất đồng bộ: trong lúc GPU bận rộn tính toán mô hình cho batch hiện tại, CPU sẽ đi đọc từ ổ cứng và xử lý trước 2 batch tiếp theo để sẵn sàng đưa cho GPU, tránh hiện tượng GPU bị 'nghẽn' do chờ dữ liệu.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Kiến trúc RetinaNet giải quyết điểm yếu nào của YOLO thế hệ đầu tiên trong việc phát hiện vật thể?",
    options: [
      "YOLO không thể xử lý ảnh màu, trong khi RetinaNet hỗ trợ định dạng RGB.",
      "YOLO kém trong việc tìm ra các vật thể quá nhỏ do lấy đặc trưng từ lớp cuối cùng có độ phân giải thấp.",
      "YOLO bắt buộc phải sử dụng phần cứng của hãng Apple, trong khi RetinaNet chạy đa nền tảng.",
      "YOLO chỉ có khả năng nhận dạng văn bản dạng chữ in, RetinaNet nhận dạng được chữ viết tay."
    ],
    correctAnswer: 1,
    explanation: "Đầu ra cuối cùng của ConvNet (dùng trong YOLO v1) có kích thước rất nhỏ (độ phân giải thấp, ví dụ 14x14). Điều này khiến nó mất đi thông tin về các vật thể bé. RetinaNet dùng FPN để kết hợp đặc trưng đa tỷ lệ, giải quyết vấn đề này.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Feature Pyramid Network (FPN) trong RetinaNet hoạt động theo nguyên lý cơ bản nào?",
    options: [
      "Kết hợp thông tin độ phân giải cao (lớp đầu) và ngữ nghĩa cao (lớp cuối) thông qua kết nối bên (lateral connections).",
      "Áp dụng cấu trúc dạng cây nhị phân để loại bỏ các vùng hình ảnh không chứa vật thể với tốc độ cực nhanh.",
      "Sử dụng kỹ thuật nén hình ảnh thành định dạng vector toán học thay vì giữ dạng lưới điểm ảnh.",
      "Loại bỏ hoàn toàn các lớp Convolutional và thay thế chúng bằng kiến trúc Transformer như mô hình ngôn ngữ."
    ],
    correctAnswer: 0,
    explanation: "FPN lấy bản đồ đặc trưng giàu ngữ nghĩa (nhưng độ phân giải thấp) ở cuối mạng, upsample nó lên, và cộng (sum) với bản đồ đặc trưng có độ phân giải cao (nhưng ít ngữ nghĩa) từ các lớp trước đó. Kết quả là tạo ra đặc trưng tốt cho mọi kích cỡ vật thể.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Tại sao máy dò một giai đoạn (như RetinaNet) lại hoạt động tốt với các dữ liệu 'out-of-distribution' (ví dụ: tranh vẽ nghệ thuật, ảnh hoạt hình)?",
    options: [
      "Vì nó được thiết kế độc quyền để phân tích hội họa thay vì ảnh chụp đời thực.",
      "Vì nó sử dụng các bộ lọc màu đặc biệt chỉ kích hoạt khi gặp các mảng màu vẽ.",
      "Vì nó nhìn nhận toàn bộ cấu trúc bức ảnh, thay vì phải tập trung phân loại các bản vá (patches) nhỏ rời rạc như R-CNN.",
      "Vì nó bỏ qua tất cả thông số màu sắc và chỉ dựa vào đường viền đen trắng của vật thể."
    ],
    correctAnswer: 2,
    explanation: "Máy dò hai giai đoạn như R-CNN phân loại từng vùng ảnh nhỏ (patch) bị cắt rời. Nếu mảng màu của bức tranh lạ lẫm, bộ phân loại sẽ thất bại. Trong khi đó máy dò một giai đoạn (YOLO, RetinaNet) dùng mạng Convolutional quét toàn cảnh, nên chúng nắm bắt được cấu trúc tổng thể (Global context) tốt hơn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Thứ tự nào sau đây biểu diễn chu trình hoạt động của Faster R-CNN (Kiến trúc hai giai đoạn)?",
    options: [
      "Đưa ảnh vào mạng, quét toàn bộ và xuất trực tiếp hộp giới hạn lẫn nhãn lớp trong một bước.",
      "Trích xuất đặc trưng (CNN) -> Mạng đề xuất vùng (RPN) -> Phân loại và Tinh chỉnh hộp giới hạn.",
      "Chia ảnh thành lưới (Grid) -> Tính điểm tin cậy từng ô -> Dự đoán lớp tương ứng.",
      "Tạo ra mạng kim tự tháp tính năng (FPN) -> Chạy cơ chế chú ý (Attention) -> Phát hiện."
    ],
    correctAnswer: 1,
    explanation: "Chu trình Faster R-CNN: Ảnh đi qua ConvNet lấy đặc trưng (Feature map). Feature map được đưa vào RPN (Region Proposal Network) để sinh các hộp đề xuất. Các hộp này tiếp tục được một mạng khác phân loại (Classification) và bóp lại cho khít (Bounding box regression).",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Thuật ngữ 'Heuristic model' (mô hình phỏng đoán/chấp nhận được) thường ám chỉ điều gì trong Machine Learning?",
    options: [
      "Một mạng nơ-ron học sâu hoàn chỉnh được huấn luyện qua hàng ngàn thế hệ epoch.",
      "Một phương pháp dựa trên một tập hợp các quy tắc do con người thiết lập mã cứng (hard-coded rules).",
      "Một thuật toán lượng tử chạy trên các máy tính siêu phân luồng tiên tiến nhất.",
      "Một tập hợp các phép tính vi phân và tích phân cấp cao."
    ],
    correctAnswer: 1,
    explanation: "Heuristic (Kinh nghiệm/Cảm tính/Quy tắc ngón tay cái) trong Machine Learning là phương pháp mà các kỹ sư viết ra các luật (rules) cứng dựa trên hiểu biết của họ (ví dụ thuật toán Selective Search), trái ngược với việc để cho mạng nơ-ron tự học luật (Learned models).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong YOLO, nếu một vật thể có kích thước cực lớn chiếm trọn gần hết bức ảnh, mạng nơ-ron xử lý điều này như thế nào?",
    options: [
      "Nó sẽ tự động chia vật thể đó ra thành hàng trăm vật thể nhỏ liti và đếm sai lệch.",
      "Lớp Flatten cho phép toàn bộ bản đồ đặc trưng có thể giao tiếp với nhau, gỡ bỏ giới hạn về không gian cục bộ.",
      "Nó sẽ báo lỗi tràn bộ nhớ do không thể vẽ một hộp giới hạn quá giới hạn cho phép.",
      "Nó sẽ tự động thu nhỏ độ phân giải hình ảnh gốc xuống mức tối thiểu trước khi tính toán."
    ],
    correctAnswer: 1,
    explanation: "Do YOLO sử dụng lớp Flatten trước khi kết nối vào các lớp Dense (Full Connected layers), mỗi ô vuông trong lưới (grid cell) đầu ra đều có khả năng tiếp cận với toàn bộ thông tin (global features) của cả bức ảnh, giúp chúng dễ dàng bao quát các vật thể lớn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "KerasHub hỗ trợ tải các mô hình phát hiện đối tượng được huấn luyện trước thông qua class nào?",
    options: [
      "keras_hub.models.TextGenerator",
      "keras_hub.models.ImageSegmenter",
      "keras_hub.models.AudioClassifier",
      "keras_hub.models.ObjectDetector"
    ],
    correctAnswer: 3,
    explanation: "Trong chương 12, KerasHub cung cấp `keras_hub.models.ObjectDetector.from_preset()` để tải mô hình phát hiện đối tượng, đi kèm luôn cả bộ tiền xử lý hình ảnh (preprocessor) tương ứng.",
    difficulty: "Dễ"
  },
  {
    type: "fill",
    question: "Phát hiện đối tượng (Object Detection) tạo ra một {1} (Bounding Box) có dạng hình chữ nhật bao quanh vật thể và gán một nhãn cho nó.",
    blanks: [
      { id: 1, text: "hộp giới hạn", answer: "hộp giới hạn|bounding box|khung giới hạn" }
    ],
    explanation: "Kết quả của Object Detection là các Bounding Box (Hộp giới hạn) xác định vị trí và kích thước của các vật thể trong ảnh.",
    difficulty: "Dễ"
  },
  {
    type: "fill",
    question: "Phương pháp tạo đặc trưng đa tỷ lệ của mô hình RetinaNet được gọi là {1} (Feature Pyramid Network).",
    blanks: [
      { id: 1, text: "mạng kim tự tháp đặc trưng", answer: "mạng kim tự tháp đặc trưng|mạng kim tự tháp tính năng|feature pyramid network|fpn" }
    ],
    explanation: "Feature Pyramid Network (FPN) là một kiến trúc giúp cải thiện khả năng phát hiện vật thể ở nhiều kích thước khác nhau, đặc biệt là các vật thể nhỏ.",
    difficulty: "Trung bình"
  },
  {
    type: "sorting",
    question: "Sắp xếp luồng xử lý của mô hình YOLO theo thứ tự đúng:",
    steps: [
      "Đưa hình ảnh đầu vào (Ví dụ: 448x448 RGB) qua bộ mã hóa xương sống (Backbone ConvNet).",
      "Trích xuất bản đồ đặc trưng cấp cao (Feature map) có độ phân giải thấp hơn.",
      "Duỗi phẳng (Flatten) bản đồ đặc trưng và truyền qua mạng kết nối đầy đủ (Dense layers).",
      "Định hình lại (Reshape) dữ liệu về dạng lưới (Grid) để dự đoán Hộp (Box) và Lớp (Class)."
    ],
    explanation: "Kiến trúc YOLO cơ bản: Backbone nén ảnh -> Lấy đặc trưng -> Flatten & Dense (xử lý toàn cục) -> Reshape thành Grid (ví dụ 6x6) chứa tọa độ Box và xác suất Class.",
    difficulty: "Khó"
  },
  {
    type: "matching",
    question: "Ghép nối các thông số dự đoán của một bounding box YOLO với ý nghĩa của nó:",
    pairs: [
      { left: "Tọa độ x, y", right: "Biểu diễn vị trí trung tâm của hộp, tương đối so với giới hạn của ô lưới." },
      { left: "Kích thước w, h", right: "Biểu diễn chiều rộng và chiều cao của hộp, tương đối so với kích thước toàn bộ ảnh." },
      { left: "Confidence score", right: "Mức độ tự tin mô hình cho rằng có vật thể và độ chính xác của hộp so với thực tế (IoU)." },
      { left: "Class probability", right: "Xác suất hộp giới hạn đó thuộc về một danh mục cụ thể (ví dụ: Chó, Ô tô, Cái cốc)." }
    ],
    explanation: "Mô hình YOLO trả về bộ 5 giá trị cho hộp (x, y, w, h, conf) cộng thêm xác suất phân loại (class) tại mỗi ô lưới.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Ghép nối các họ mô hình phát hiện đối tượng với đặc tính của chúng:",
    pairs: [
      { left: "R-CNN (Region-based CNN)", right: "Hai giai đoạn: Đề xuất vùng bằng thuật toán cứng (Selective Search) rồi mới phân loại." },
      { left: "Faster R-CNN", right: "Hai giai đoạn: Dùng chính mạng nơ-ron (RPN) để đề xuất vùng, nhanh hơn R-CNN cũ." },
      { left: "YOLO (You Only Look Once)", right: "Một giai đoạn: Quét ảnh một lần duy nhất, tốc độ cực nhanh, hay dùng cho Real-time." },
      { left: "RetinaNet", right: "Một giai đoạn: Sử dụng FPN (Feature Pyramid) để cải thiện độ chính xác cho vật thể nhỏ." }
    ],
    explanation: "Sự phân loại rõ ràng giữa các dòng máy dò phát hiện đối tượng phổ biến được nêu trong chương 12.",
    difficulty: "Trung bình"
  }
];

export default questions;
