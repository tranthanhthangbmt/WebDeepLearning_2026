const questions = [
  {
    type: "mcq",
    question: "Tác vụ nào sau đây KHÔNG thuộc về ba nhiệm vụ cơ bản nhất của thị giác máy tính?",
    options: [
      "Phân loại hình ảnh (Image Classification).",
      "Phân đoạn hình ảnh (Image Segmentation).",
      "Phát hiện đối tượng (Object Detection).",
      "Tái tạo âm thanh không gian (Spatial Audio)."
    ],
    correctAnswer: 3,
    explanation: "Ba tác vụ cốt lõi của thị giác máy tính bao gồm: Phân loại hình ảnh (Classification), Phân đoạn hình ảnh (Segmentation) và Phát hiện đối tượng (Detection). Tái tạo âm thanh không thuộc lĩnh vực này.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Điểm khác biệt lớn nhất giữa phân loại hình ảnh (Classification) và phát hiện đối tượng (Object Detection) là gì?",
    options: [
      "Phân loại hình ảnh chỉ dùng được cho ảnh đen trắng, còn phát hiện đối tượng dùng cho ảnh màu RGB.",
      "Phân loại chỉ gán nhãn cho toàn bộ bức ảnh, còn phát hiện đối tượng vẽ hộp giới hạn (bounding box).",
      "Phân loại hình ảnh luôn chạy trên CPU, trong khi phát hiện đối tượng bắt buộc phải dùng GPU.",
      "Phân loại chỉ có thể nhận diện chó và mèo, trong khi phát hiện đối tượng có thể nhận diện biển báo."
    ],
    correctAnswer: 1,
    explanation: "Phân loại gán nhãn chung cho cả bức ảnh (ví dụ: 'có mèo'). Phát hiện đối tượng chỉ ra vị trí chính xác của đối tượng đó bằng cách vẽ một hình chữ nhật (bounding box) bao quanh nó.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Phân đoạn hình ảnh (Image Segmentation) nhằm giải quyết bài toán nào dưới đây ở cấp độ chi tiết nhất?",
    options: [
      "Tăng kích thước và độ phân giải hình ảnh gốc.",
      "Gán một nhãn phân loại cho toàn bộ hình ảnh.",
      "Gán nhãn phân loại cho TỪNG pixel trong ảnh.",
      "Vẽ một hộp hình chữ nhật bao quanh đối tượng."
    ],
    correctAnswer: 2,
    explanation: "Phân đoạn hình ảnh (Segmentation) có độ phân giải chi tiết nhất vì nó yêu cầu mạng nơ-ron phải dự đoán chính xác lớp (class) cho TỪNG pixel riêng lẻ trong bức ảnh đầu vào.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Loại phân đoạn nào sẽ gán CÙNG MỘT nhãn cho tất cả các con mèo xuất hiện trong một bức ảnh (ví dụ: gộp chung thành 'cat')?",
    options: [
      "Phân đoạn thực thể (Instance Segmentation).",
      "Phân đoạn ngữ nghĩa (Semantic Segmentation).",
      "Phân đoạn toàn cảnh (Panoptic Segmentation).",
      "Phân đoạn hình học (Geometric Segmentation)."
    ],
    correctAnswer: 1,
    explanation: "Phân đoạn ngữ nghĩa (Semantic Segmentation) chỉ quan tâm pixel đó thuộc danh mục nào. Nếu có 2 con mèo, tất cả pixel của chúng đều được gán nhãn 'mèo' mà không phân biệt con số 1 hay con số 2.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Phân đoạn toàn cảnh (Panoptic Segmentation) là sự kết hợp của hai kỹ thuật nào?",
    options: [
      "Phân đoạn ngữ nghĩa (Semantic) VÀ Phân đoạn phiên bản (Instance).",
      "Phân loại hình ảnh (Classification) VÀ Phát hiện đối tượng (Detection).",
      "Xử lý ngôn ngữ tự nhiên (NLP) VÀ Phân đoạn ngữ nghĩa (Semantic).",
      "Tạo sinh hình ảnh (Generative) VÀ Phân đoạn điểm ảnh (Pixel-wise)."
    ],
    correctAnswer: 0,
    explanation: "Phân đoạn toàn cảnh (Panoptic) kết hợp Semantic (nhãn lớp: 'mèo', 'trời') và Instance (số phiên bản: 'con mèo số 1', 'con mèo số 2'), mang lại lượng thông tin đầy đủ nhất.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong tập dữ liệu Oxford-IIIT Pets (phân đoạn Semantic), mặt nạ mục tiêu (mask) thường có mấy giá trị nguyên cho mỗi pixel?",
    options: [
      "Hàng triệu giá trị tương ứng phổ màu RGB.",
      "Chỉ hai giá trị duy nhất là số 0 và số 1.",
      "Ba giá trị: Tiền cảnh, hậu cảnh, đường viền.",
      "Mười giá trị vì có mười giống chó mèo khác nhau."
    ],
    correctAnswer: 2,
    explanation: "Trong tập dữ liệu này, mặt nạ phân đoạn chia thành 3 lớp (classes) cho từng pixel: 1 (foreground - thú cưng), 2 (background - nền), 3 (contour - đường viền).",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Khi xây dựng nửa đầu của mô hình phân đoạn (mã hóa), tại sao tác giả khuyên DÙNG strided convolutions thay vì MaxPooling2D?",
    options: [
      "Vì MaxPooling2D làm tăng số lượng tham số khiến mô hình chậm chạp.",
      "Vì strided convolutions giữ lại thông tin vị trí không gian tốt hơn.",
      "Vì MaxPooling2D chỉ chạy được trên CPU, không tận dụng được sức mạnh GPU.",
      "Vì strided convolutions tự động chuyển ảnh màu thành ảnh đen trắng đen."
    ],
    correctAnswer: 1,
    explanation: "Đối với Segmentation, vị trí từng pixel (không gian) rất quan trọng. MaxPooling hủy hoại thông tin vị trí cục bộ trong mỗi cửa sổ 2x2. Dùng sải bước (strides) giúp nén đặc trưng mà vẫn giữ được vị trí tương đối tốt hơn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Chức năng của các lớp Conv2DTranspose trong kiến trúc phân đoạn hình ảnh là gì?",
    options: [
      "Thu nhỏ (downsample) hình ảnh để loại bỏ các chi tiết nhiễu không cần thiết.",
      "Tính toán trực tiếp giá trị hàm mất mát IoU (Intersection over Union).",
      "Mở rộng (upsample) bản đồ đặc trưng trở lại kích thước hình ảnh đầu vào ban đầu.",
      "Tự động trích xuất các khung văn bản (bounding boxes) ra khỏi hình ảnh."
    ],
    correctAnswer: 2,
    explanation: "Trong mô hình hình phễu (Encoder-Decoder) của bài toán phân đoạn, Conv2DTranspose đóng vai trò như một Decoder: nó học cách 'mở rộng' (upsample) tensor đặc trưng nhỏ gọn trở lại kích thước ảnh gốc để xuất ra mask.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Tại sao trong mô hình phân đoạn (Segmentation), lớp đầu ra cuối cùng thường sử dụng hàm kích hoạt Softmax?",
    options: [
      "Để nén giá trị độ sáng của các pixel trong ảnh xuống còn đúng một con số.",
      "Để biến mô hình từ xử lý hình ảnh sang xử lý chuỗi văn bản tự nhiên NLP.",
      "Để dự đoán xác suất mỗi pixel thuộc về một trong ba lớp riêng biệt.",
      "Để loại bỏ các pixel màu đen ra khỏi quá trình tính toán của mô hình."
    ],
    correctAnswer: 2,
    explanation: "Bởi vì ta cần phân loại độc lập cho *từng pixel* vào một trong 3 nhóm (Tiền cảnh, Hậu cảnh, Viền). Softmax phân bổ xác suất cho 3 lớp này, lớp có xác suất cao nhất sẽ là nhãn dự đoán cho pixel đó.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Chỉ số IoU (Intersection over Union) thường được sử dụng để đánh giá điều gì trong Computer Vision?",
    options: [
      "Tốc độ xử lý (frames per second) của mạng trên phần cứng GPU tiêu chuẩn.",
      "Số lượng tham số (parameters) cần thiết để huấn luyện thành công mô hình.",
      "Mức độ trùng khớp giữa mặt nạ dự đoán và mặt nạ sự thật cơ bản (ground truth).",
      "Khả năng loại bỏ nhiễu của hình ảnh khi chụp trong điều kiện ban đêm thiếu sáng."
    ],
    correctAnswer: 2,
    explanation: "IoU là tiêu chuẩn vàng cho bài toán Phân đoạn (và Phát hiện đối tượng). Nó lấy 'Phần giao nhau' (diện tích trùng khớp) chia cho 'Phần hợp' (tổng diện tích) của hai mặt nạ để đánh giá chất lượng dự đoán.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong công thức tính IoU, nếu dự đoán hoàn toàn sai lệch (không có pixel nào trùng khớp), giá trị IoU sẽ bằng bao nhiêu?",
    options: [
      "Luôn luôn bằng giá trị 1.0",
      "Sẽ tiến tới dương vô cực.",
      "Luôn luôn bằng giá trị 0.0",
      "Sẽ rơi vào số âm (khoảng -1)."
    ],
    correctAnswer: 2,
    explanation: "Nếu không có sự trùng lặp (Intersection = 0), thì IoU = 0 / Union = 0. Giá trị IoU lý tưởng là 1.0 (trùng khớp 100%).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong ví dụ ở Chương 11, mô hình phân đoạn sử dụng hàm loss nào khi biên dịch (compile)?",
    options: [
      "Mean Squared Error (MSE)",
      "Binary Crossentropy",
      "Sparse Categorical Crossentropy",
      "Huber Loss"
    ],
    correctAnswer: 2,
    explanation: "Vì mặt nạ sự thật (ground truth mask) chứa các nhãn dạng số nguyên (integer labels) thay vì one-hot vectors, và ta có > 2 lớp (3 lớp: background, foreground, border), nên dùng Sparse Categorical Crossentropy.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Mô hình SAM (Segment Anything Model) được phát triển bởi công ty/tổ chức nào?",
    options: [
      "Google Research",
      "Meta AI (Facebook)",
      "OpenAI",
      "Microsoft Research"
    ],
    correctAnswer: 1,
    explanation: "SAM là một bước đột phá trong thị giác máy tính do Meta AI giới thiệu vào tháng 4 năm 2023.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Đặc điểm nổi bật nhất (innovation) của mô hình SAM so với các mô hình phân đoạn truyền thống là gì?",
    options: [
      "Nó là mô hình phân đoạn đầu tiên có thể chạy được trên điện thoại không cần internet.",
      "Nó không bị giới hạn ở các lớp định trước, có thể phân đoạn vật thể lạ bằng 'Prompt'.",
      "Nó loại bỏ hoàn toàn các lớp Convolution và chỉ dùng các lớp Dense kết nối đầy đủ.",
      "Nó là mô hình duy nhất không cần sử dụng bất kỳ dữ liệu huấn luyện nào trước đó."
    ],
    correctAnswer: 1,
    explanation: "SAM là mô hình 'zero-shot' linh hoạt. Giống như ChatGPT, bạn không cần phải fine-tune nó cho vật thể mới. Bạn chỉ cần cung cấp Prompt (dấu nhắc điểm, hoặc hộp) và nó sẽ phân đoạn bất kỳ thứ gì.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Tập dữ liệu SA-1B được dùng để huấn luyện mô hình SAM có quy mô khoảng bao nhiêu?",
    options: [
      "100 ngàn hình ảnh với 1 triệu mặt nạ.",
      "1 triệu hình ảnh với 10 triệu mặt nạ.",
      "11 triệu hình ảnh với hơn 1 tỷ mặt nạ.",
      "100 triệu hình ảnh mà không có mặt nạ."
    ],
    correctAnswer: 2,
    explanation: "SA-1B là tập dữ liệu phân đoạn lớn nhất thế giới, chứa 11 triệu hình ảnh và hơn 1 tỷ mặt nạ phân đoạn (trung bình ~100 mặt nạ mỗi ảnh).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong quy trình xây dựng tập dữ liệu SA-1B, kỹ thuật 'Data Engine' (vòng lặp phản hồi) hoạt động như thế nào?",
    options: [
      "Người dùng internet tải ảnh lên và hệ thống tự động gán nhãn ngẫu nhiên không kiểm soát.",
      "Mô hình yếu hỗ trợ con người gán nhãn nhanh hơn, lấy dữ liệu đó để luyện lại thành mô hình mạnh hơn.",
      "Sử dụng công cụ đồ họa 3D để render hình ảnh máy tính và dùng chúng làm tập huấn luyện duy nhất.",
      "Thuê hàng triệu người dùng chỉ để khoanh vùng (bounding box) mà không cần phân đoạn pixel."
    ],
    correctAnswer: 1,
    explanation: "Meta sử dụng phương pháp phát triển song song: Chuyên gia gán nhãn -> Huấn luyện mô hình -> Dùng mô hình tự động phân đoạn sơ bộ -> Con người chỉnh sửa lại -> Đưa vào huấn luyện lại (Feedback loop).",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Để sử dụng mô hình SAM trong thực tế, bạn cần cung cấp một đầu vào là hình ảnh và yếu tố thứ hai là gì?",
    options: [
      "Một đoạn mã nguồn C++ điều khiển tiến trình GPU.",
      "Một Dấu nhắc (Prompt) như điểm (point) hoặc hộp (box).",
      "Một tệp tin cấu hình siêu tham số (hyperparameters).",
      "Một mặt nạ sự thật cơ bản (ground truth) có độ phân giải cao."
    ],
    correctAnswer: 1,
    explanation: "Mô hình SAM yêu cầu đầu vào là bộ đôi: Hình ảnh (Image) + Dấu nhắc (Prompt). Dấu nhắc có thể là một điểm click chuột (point), một khung bao quanh (box), hoặc văn bản (text).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Khi cung cấp Dấu nhắc Điểm (Point prompt) cho mô hình SAM, giá trị nhãn (label) = 1 có ý nghĩa gì?",
    options: [
      "Điểm đó là phần nền (Background) mà bạn không muốn chọn.",
      "Điểm đó là đối tượng tiền cảnh (Foreground) bạn muốn phân đoạn.",
      "Điểm đó là vùng nhiễu (Noise) cần thuật toán loại bỏ hoàn toàn.",
      "Điểm đó đại diện cho viền (Contour) của bức ảnh tổng thể."
    ],
    correctAnswer: 1,
    explanation: "Khi truyền điểm vào SAM, `input_label = [1]` báo cho mô hình biết đây là đối tượng mục tiêu. `input_label = [0]` báo cho mô hình biết điểm đó là phần nền cần loại trừ.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Kiến trúc mô hình SAM bao gồm 3 thành phần chính nào?",
    options: [
      "Bộ mã hóa hình ảnh (Image encoder), Bộ mã hóa dấu nhắc (Prompt encoder), Bộ giải mã mặt nạ (Mask decoder).",
      "Mạng Recurrent Neural Network (RNN), Mạng Bộ nhớ LSTM, Mạng kết nối đầy đủ (Dense Network).",
      "Bộ nén video (Video compressor), Bộ cân bằng màu (Color equalizer), Mạng tạo sinh GAN (Generator).",
      "Bộ trích xuất viền (Edge detector), Bộ lọc nhiễu Gauss (Gaussian filter), Bộ phân loại SVM (Classifier)."
    ],
    correctAnswer: 0,
    explanation: "SAM chia thành 3 khối rõ rệt: Image encoder (tính toán nặng, trích xuất đặc trưng ảnh một lần duy nhất), Prompt encoder (mã hóa các tọa độ/hộp), và Mask decoder (nhẹ, trộn ảnh và dấu nhắc để sinh mask).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Kích thước đầu vào chuẩn (default input size) mà mô hình SAM yêu cầu trước khi xử lý là bao nhiêu?",
    options: [
      "224 x 224 pixel.",
      "256 x 256 pixel.",
      "1024 x 1024 pixel.",
      "Giữ nguyên kích thước gốc không giới hạn."
    ],
    correctAnswer: 2,
    explanation: "SAM kỳ vọng kích thước đầu vào là 1024x1024. Nếu ảnh không vuông, ta phải thay đổi kích thước sao cho cạnh dài nhất bằng 1024 và thêm khoảng đệm (padding zeros) vào để giữ tỷ lệ khung hình.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Làm thế nào để thay đổi kích thước ảnh không vuông thành 1024x1024 cho SAM mà không làm méo hình?",
    options: [
      "Ép hình ảnh dãn dài ra (squash) bất chấp tỷ lệ gốc ban đầu.",
      "Cắt bỏ phần thừa ở hai cạnh (crop) để lấy trung tâm bức ảnh.",
      "Thay đổi cạnh dài nhất thành 1024 và đệm viền đen (pad_to_aspect_ratio).",
      "Chia ảnh thành nhiều phần vuông nhỏ và chạy qua mô hình nhiều lần."
    ],
    correctAnswer: 2,
    explanation: "Việc ép tỷ lệ (squashing) sẽ phá hỏng hình khối thực tế. Giải pháp tốt nhất là resize giữ nguyên tỷ lệ khung hình (`pad_to_aspect_ratio=True`), sau đó điền các viền đen (zero padding).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong kết quả trả về của SAM, mảng 'masks' có kích thước (1, 4, 256, 256). Con số '4' ở đây đại diện cho điều gì?",
    options: [
      "Bốn kênh màu Red, Green, Blue và Alpha (độ trong suốt) của bức ảnh.",
      "Bốn mặt nạ ứng cử viên (candidate masks) khác nhau để giải quyết sự mơ hồ của lời nhắc.",
      "Bốn góc của khung giới hạn hình chữ nhật (bounding box) bao quanh đối tượng mục tiêu.",
      "Bốn lớp (classes) cơ bản nhất mà SAM luôn phân loại mặt định cho mọi hình ảnh."
    ],
    correctAnswer: 1,
    explanation: "Một lời nhắc điểm (point prompt) có thể mơ hồ. Ví dụ chấm vào áo sơ mi của người, SAM sẽ trả về 4 mask: (1) cái cúc áo, (2) cái áo sơ mi, (3) người đàn ông, để người dùng tự chọn mask ưng ý nhất.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Mô hình SAM phiên bản `sam_huge_sa1b` có khoảng bao nhiêu tham số (parameters)?",
    options: [
      "Dưới 1 triệu tham số.",
      "Khoảng 25 triệu tham số.",
      "Khoảng 641 triệu tham số.",
      "Hơn 100 tỷ tham số."
    ],
    correctAnswer: 2,
    explanation: "Theo sách, `sam_huge_sa1b` có chính xác `641090864` (khoảng 641 triệu) tham số, là mô hình nặng nhất trong cuốn sách tính đến thời điểm này.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Khi sử dụng lời nhắc hộp (Box prompt) cho SAM, dữ liệu truyền vào là gì?",
    options: [
      "Toạ độ tâm của hộp và giá trị bán kính đường tròn nội tiếp.",
      "Toạ độ góc trên cùng bên trái và góc dưới cùng bên phải của hộp.",
      "Một mảng chứa toạ độ của cả bốn góc tứ giác cùng một lúc.",
      "Mã màu của hộp (RGB) để mô hình tìm kiếm trên toàn bộ bức ảnh."
    ],
    correctAnswer: 1,
    explanation: "Dấu nhắc hộp (Box prompt) yêu cầu một mảng 2 điểm ảnh: góc trên bên trái `[x_min, y_min]` và góc dưới bên phải `[x_max, y_max]`.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Tại sao trong mô hình phân đoạn hình ảnh Oxford-IIIT Pets, ta lại dùng hàm loss `sparse_categorical_crossentropy`?",
    options: [
      "Vì ta chỉ đang thực hiện phân loại nhị phân (có/không có con vật).",
      "Vì tập nhãn là các số nguyên 0,1,2 đại diện cho 3 lớp (tiền cảnh, nền, viền).",
      "Vì ta muốn đo lường trực tiếp sự trùng lặp diện tích của IoU.",
      "Vì các nhãn mục tiêu (targets) chứa số thực thập phân có dấu."
    ],
    correctAnswer: 1,
    explanation: "Bất cứ khi nào nhãn mục tiêu (targets) của bài toán phân loại nhiều lớp (từ 3 lớp trở lên) được mã hóa dưới dạng số nguyên (integer labels: 0, 1, 2) thay vì one-hot encoding, ta bắt buộc phải dùng `sparse_categorical_crossentropy`.",
    difficulty: "Trung bình"
  },
  {
    type: "fill",
    question: "Tác vụ {1} (Image segmentation) phân vùng hình ảnh thành nhiều khu vực khác nhau bằng cách phân loại từng điểm ảnh riêng lẻ.",
    blanks: [
      { id: 1, text: "phân đoạn hình ảnh", answer: "phân đoạn hình ảnh|phân đoạn|image segmentation" }
    ],
    explanation: "Phân đoạn hình ảnh (Image segmentation) là bài toán gán nhãn cho mọi điểm ảnh trong bức ảnh, cao cấp và chi tiết hơn nhiều so với Phân loại hình ảnh.",
    difficulty: "Dễ"
  },
  {
    type: "fill",
    question: "Để giải quyết sự mơ hồ, mô hình SAM trả về nhiều mặt nạ ứng cử viên được xếp hạng bởi điểm số {1} dự đoán cho mỗi mặt nạ.",
    blanks: [
      { id: 1, text: "iou", answer: "iou|intersection over union" }
    ],
    explanation: "Điểm IoU (Intersection over Union) pred dùng để xếp hạng độ tin cậy của các candidate mask do SAM tạo ra.",
    difficulty: "Trung bình"
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự một quy trình hoạt động của mô hình phân đoạn (Encoder-Decoder):",
    steps: [
      "Nhận hình ảnh RGB làm đầu vào.",
      "Sử dụng các lớp Conv2D kết hợp sải bước (Strides) để nén ảnh thành bản đồ đặc trưng nhỏ.",
      "Sử dụng các lớp Conv2DTranspose để mở rộng (upsample) đặc trưng trở lại kích thước cũ.",
      "Áp dụng hàm Softmax ở lớp Conv2D cuối để dự đoán xác suất từng pixel."
    ],
    explanation: "Kiến trúc chung (U-Net style hoặc Encoder-Decoder): Vào -> Thu nhỏ (Downsample bằng Strides) -> Phóng to (Upsample bằng Transpose) -> Phân loại từng pixel bằng Softmax.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Ghép nối các tác vụ Computer Vision với ứng dụng thực tế đặc trưng nhất:",
    pairs: [
      { left: "Image Classification", right: "Gắn thẻ thư viện ảnh Google Photos (VD: ảnh này có 'cún')" },
      { left: "Image Segmentation", right: "Xóa phông nền hoặc thay nền ảo trong Zoom/Google Meet" },
      { left: "Object Detection", right: "Xe tự lái khoanh vùng các biển báo giao thông và người đi bộ" },
      { left: "Keypoint Detection", right: "Phát hiện các điểm mốc trên khuôn mặt (mắt, mũi, miệng) để làm filter AR" }
    ],
    explanation: "Đây là các ví dụ trực quan nhất về sự khác biệt giữa các nhánh chính của Computer Vision được đề cập đầu Chương 11.",
    difficulty: "Dễ"
  },
  {
    type: "matching",
    question: "Ghép nối các loại phân đoạn hình ảnh (Image Segmentation) với tính chất của chúng:",
    pairs: [
      { left: "Semantic segmentation", right: "Không phân biệt giữa nhiều đối tượng cùng một loài (Gộp chung thành 1 nhãn)" },
      { left: "Instance segmentation", right: "Tách biệt rõ ràng từng cá thể riêng biệt trong cùng một bức ảnh" },
      { left: "Panoptic segmentation", right: "Gán nhãn ngữ nghĩa VÀ nhãn cá thể cho mọi pixel (đầy đủ thông tin nhất)" }
    ],
    explanation: "Định nghĩa chính xác về 3 biến thể phân đoạn được thảo luận trong phần đầu chương 11.",
    difficulty: "Trung bình"
  }
];

export default questions;
