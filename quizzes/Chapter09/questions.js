const questions = [
  {
    type: "mcq",
    question: "Công thức MHR (Modularity, Hierarchy, Reuse) dùng để giải quyết vấn đề gì trong kiến trúc hệ thống?",
    options: [
      "Tăng cường độ sáng của hình ảnh đầu vào thông qua các bộ lọc tự động.",
      "Cấu trúc một hệ thống phức tạp trở nên đơn giản và dễ quản lý hơn.",
      "Chuyển đổi dữ liệu chuỗi thành hình ảnh hai chiều thông qua hàm băm.",
      "Loại bỏ các đặc trưng không quan trọng bằng cách dùng pooling liên tục."
    ],
    correctAnswer: 1,
    explanation: "Công thức MHR (Mô-đun - Phân cấp - Tái sử dụng) là nguyên tắc cốt lõi để làm cho một hệ thống phức tạp trở nên đơn giản, chia nhỏ thành các mô-đun phân cấp và tái sử dụng chúng (như việc cấu trúc các lớp trong Keras).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong Học sâu, tính 'Tái sử dụng' (Reuse) được thể hiện rõ nhất qua phép toán nào sau đây?",
    options: [
      "Kích hoạt phi tuyến (ReLU) vì nó chỉ loại bỏ phần âm của dữ liệu.",
      "Tích chập (Convolution) vì nó dùng chung trọng số ở mọi vị trí.",
      "Lớp kết nối đầy đủ (Dense) vì nó liên kết toàn bộ tham số vào mọi node.",
      "Hàm tính lỗi (Loss function) vì nó đánh giá toàn bộ kết quả tổng quát."
    ],
    correctAnswer: 1,
    explanation: "Phép tích chập là sự tái sử dụng thông tin mạnh mẽ nhất, nơi mà cùng một bộ lọc (filter/kernel) được áp dụng ở tất cả các vị trí không gian khác nhau trên ảnh.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Quy luật nào sau đây thường được áp dụng cho số lượng bộ lọc (filters) trong một kiến trúc ConvNet phân cấp?",
    options: [
      "Số lượng bộ lọc tăng dần khi kích thước của bản đồ tính năng giảm xuống.",
      "Số lượng bộ lọc giảm đi một nửa sau mỗi lớp tích chập theo thứ tự.",
      "Giữ nguyên số lượng bộ lọc không thay đổi trong toàn bộ mô hình.",
      "Số lượng bộ lọc được quyết định ngẫu nhiên trong mỗi lần khởi tạo."
    ],
    correctAnswer: 0,
    explanation: "Cấu trúc kim tự tháp (feature hierarchies) rất phổ biến: khi đi sâu vào mạng, không gian không gian (chiều rộng/chiều cao) co lại do MaxPooling, trong khi chiều sâu (số channels/filters) tăng lên để biểu diễn các đặc trưng phức tạp hơn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Một ngăn xếp (stack) sâu gồm các lớp hẹp so với một ngăn xếp nông gồm các lớp lớn sẽ có hiệu suất như thế nào?",
    options: [
      "Ngăn xếp nông lớn luôn vượt trội vì nó chứa lượng tham số khổng lồ.",
      "Ngăn xếp sâu hẹp thường hoạt động tốt hơn nhờ phân cấp tính năng.",
      "Cả hai luôn có chung hiệu năng nếu số lượng tổng node bằng hệt nhau.",
      "Ngăn xếp sâu hẹp không thể đào tạo được do yêu cầu bộ nhớ quá nhỏ."
    ],
    correctAnswer: 1,
    explanation: "Hệ thống phân cấp sâu hơn (deep and narrow) thường khuyến khích việc tái sử dụng tính năng và tính trừu tượng tốt hơn so với mô hình rộng nhưng nông (shallow and broad).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Nghiên cứu cắt bỏ (Ablation study) trong học sâu nhằm mục đích chính là gì?",
    options: [
      "Bổ sung thêm hàng loạt lớp ẩn phức tạp để cải thiện hiệu suất.",
      "Xóa bỏ hệ thống mạng để thay thế bằng mạng mới hoàn toàn.",
      "Loại bỏ có hệ thống từng phần để hiểu nguyên nhân tạo ra hiệu suất.",
      "Giảm dung lượng mô hình để có thể triển khai trên thiết bị di động."
    ],
    correctAnswer: 2,
    explanation: "Ablation study liên quan đến việc làm hệ thống đơn giản hơn (loại bỏ các thành phần) để hiểu quan hệ nhân quả: thành phần nào thực sự đóng góp vào hiệu năng và thành phần nào là thừa.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Hiện tượng 'Độ dốc biến mất' (Vanishing gradients) xảy ra khi nào?",
    options: [
      "Mạng quá nông khiến cho mô hình không đủ khả năng học các hàm phức tạp.",
      "Tốc độ học (learning rate) được đặt quá lớn khiến hàm mất mát bùng nổ.",
      "Mạng quá sâu khiến tín hiệu lỗi bị nhiễu và triệt tiêu qua các lớp.",
      "Sử dụng hàm kích hoạt ReLU trên các giá trị âm của biểu diễn đầu vào."
    ],
    correctAnswer: 2,
    explanation: "Khi mạng quá sâu, tín hiệu lỗi (gradient) lan truyền ngược qua chuỗi các hàm số sẽ bị suy giảm và nhiễu (giống trò chơi truyền tin nhắn), dẫn đến các lớp đầu tiên không cập nhật được trọng số.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Kỹ thuật nào sau đây được thiết kế chủ yếu để khắc phục vấn đề độ dốc biến mất ở mạng thần kinh sâu?",
    options: [
      "Tăng kích thước batch size (kích thước lô) lên mức lớn nhất.",
      "Thay thế các lớp Conv2D bằng mạng kết nối đầy đủ (Dense).",
      "Sử dụng các kết nối dư (Residual connections) giữa các khối.",
      "Tăng kích thước ảnh đầu vào (resolution) lên gấp đôi bình thường."
    ],
    correctAnswer: 2,
    explanation: "Kết nối dư (Residual connections) đóng vai trò như một lối tắt thông tin (shortcut), cho phép gradient lan truyền ngược một cách trơn tru qua các khối mà không bị suy giảm đáng kể.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Trong một kết nối dư (residual connection), thao tác toán học nào được dùng để kết hợp luồng dữ liệu chính và phần dư (residual)?",
    options: [
      "Phép nhân ma trận (Matrix multiplication) giữa tensor chính và dư.",
      "Phép cộng ma trận (Addition) giữa tensor chính và tensor dư.",
      "Nối chuỗi (Concatenation) dọc theo trục không gian của bản đồ.",
      "Phép lấy trung bình (Averaging) từng điểm ảnh của các tensor."
    ],
    correctAnswer: 1,
    explanation: "Trong Residual connection, ta thêm thẳng (cộng) tensor đầu vào (residual) vào tensor đầu ra của khối: `x = layers.add([x, residual])`.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Khi kết nối dư bắc cầu qua một khối có sự gia tăng số lượng bộ lọc (channels), ta cần làm gì với phần dư (residual) để có thể thực hiện phép cộng?",
    options: [
      "Loại bỏ phép cộng và thay bằng phép nối (concatenate) dọc trục kênh.",
      "Sử dụng lớp Conv2D 1x1 để chiếu phần dư sang số kênh tương ứng.",
      "Sao chép các kênh của phần dư lên nhiều lần để có độ sâu bằng nhau.",
      "Đệm bằng các giá trị zero (zero-padding) ở cuối tensor của phần dư."
    ],
    correctAnswer: 1,
    explanation: "Vì tensor chính có số bộ lọc lớn hơn, ta dùng phép chiếu tuyến tính (1x1 Conv2D layer không có activation) để tăng số kênh của phần dư lên cho bằng với khối đầu ra trước khi cộng.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Tại sao trong các khối có kết nối dư, ta thường dùng `padding='same'` cho các lớp Conv2D?",
    options: [
      "Để giảm thiểu số lượng phép toán dấu phẩy động cần thực hiện.",
      "Để giữ cho kích thước không gian của ảnh không bị nhỏ lại do viền.",
      "Để triệt tiêu hiện tượng overfitting gây ra bởi các điểm ảnh biên.",
      "Để tăng số lượng bộ lọc đặc trưng một cách tự động khi qua lớp."
    ],
    correctAnswer: 1,
    explanation: "Dùng `padding='same'` đảm bảo bản đồ đặc trưng (chiều rộng/cao) không bị thu nhỏ lại sau phép tích chập, nhờ đó kích thước không gian của luồng chính và phần dư hoàn toàn ăn khớp để cộng.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Chuẩn hóa dữ liệu đầu vào (Input Normalization) thường bao gồm các bước nào?",
    options: [
      "Chuyển ảnh về ảnh xám và cân bằng biểu đồ histogram của ảnh.",
      "Trừ đi giá trị trung bình và chia cho độ lệch chuẩn của dữ liệu.",
      "Nhân mọi giá trị điểm ảnh với hệ số 255 để đưa về dải chuẩn.",
      "Chuyển đổi toàn bộ mảng numpy thành các số thực dưới 0 tuyệt đối."
    ],
    correctAnswer: 1,
    explanation: "Chuẩn hóa thông thường nghĩa là đưa trung bình (mean) về 0 bằng cách trừ giá trị trung bình, và đưa phương sai (variance) về 1 bằng cách chia cho độ lệch chuẩn.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Lớp BatchNormalization trong Keras sử dụng thống kê nào trong quá trình SUY LUẬN (Inference)?",
    options: [
      "Sử dụng trung bình và phương sai của lô dữ liệu kiểm tra lúc suy luận.",
      "Sử dụng giá trị mặc định là mean=0, variance=1 ở mọi trường hợp.",
      "Sử dụng trung bình động hàm mũ (moving average) tính từ lúc huấn luyện.",
      "Sử dụng toàn bộ tham số của lớp Dense cuối cùng làm hệ số chuẩn hóa."
    ],
    correctAnswer: 2,
    explanation: "Trong lúc huấn luyện, BatchNormalization tính thống kê trên batch hiện tại. Trong lúc suy luận, không có batch lớn, nên nó sử dụng trung bình động (moving average) đã được tích lũy từ quá trình huấn luyện.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Khi sử dụng BatchNormalization ngay sau lớp Conv2D, ta có thể tối ưu hóa tham số mạng bằng cách nào?",
    options: [
      "Sử dụng hàm kích hoạt Sigmoid trong toàn bộ mô hình thay vì ReLU.",
      "Đặt tùy chọn use_bias=False trong lớp Conv2D đứng trước BatchNormalization.",
      "Giảm kích thước của kernel trong Conv2D xuống 1x1 để bù đắp tham số.",
      "Bỏ qua mọi kỹ thuật đệm (padding='valid') để loại nhiễu ranh giới ảnh."
    ],
    correctAnswer: 1,
    explanation: "BatchNormalization sẽ tự động căn giữa dữ liệu (đưa trung bình về 0), do đó việc Conv2D cộng thêm một bias vector hằng số là thừa. Ta có thể bỏ bias trong Conv2D (use_bias=False) để mạng nhẹ hơn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Điều gì sẽ xảy ra nếu để các lớp BatchNormalization có thể huấn luyện (trainable=True) trong quá trình Fine-tuning (tinh chỉnh) mô hình?",
    options: [
      "Chúng sẽ giúp mô hình tăng cường khả năng chống lại nhiễu cực đại.",
      "Chúng liên tục thay đổi phương sai nội bộ, phá vỡ các tính năng vi mô.",
      "Chúng vô hiệu hóa tất cả các lớp tích chập đứng trước trong cùng khối.",
      "Chúng làm tăng tốc độ học cục bộ lên quá mức dẫn đến vỡ gradient."
    ],
    correctAnswer: 1,
    explanation: "Khi tinh chỉnh (fine-tuning) một mô hình pre-trained, ta nên đóng băng BatchNormalization. Nếu không, thống kê trung bình/phương sai của BN liên tục thay đổi, sẽ phá hủy các cập nhật trọng số vô cùng nhỏ ở các lớp Conv2D xung quanh.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Vị trí đặt hàm kích hoạt (Activation) so với BatchNormalization đang được khuyến nghị trong Keras là gì?",
    options: [
      "Đặt BatchNormalization SAU lớp Activation (Conv -> Act -> BN).",
      "Đặt BatchNormalization TRƯỚC lớp Activation (Conv -> BN -> Act).",
      "Thay thế hoàn toàn lớp Activation bằng lớp BatchNormalization.",
      "Không quan trọng, vị trí nào cũng dẫn đến một kết quả giống hệt nhau."
    ],
    correctAnswer: 1,
    explanation: "Dù cả 2 cách đều học được, nhưng việc chuẩn hóa (BN) đầu ra của Conv2D trước rồi mới đưa qua hàm phi tuyến (như ReLU) sẽ tận dụng tối đa tính chất của ReLU (lấy 0 làm trục xoay giữ/bỏ kênh).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Lớp Tích chập có thể phân tách theo chiều sâu (Depthwise Separable Convolution) thực hiện hai phép toán nào và theo thứ tự nào?",
    options: [
      "Tích chập điểm ảnh (1x1) trước, rồi mới Tích chập không gian (3x3).",
      "Tích chập không gian độc lập trên mỗi kênh, sau đó tích chập điểm (1x1).",
      "Tích chập 3 chiều đồng thời trên mọi kênh và mọi không gian điểm ảnh.",
      "Gộp trung bình (Average Pooling) toàn cầu, sau đó tích chập chéo kênh."
    ],
    correctAnswer: 1,
    explanation: "Nó tách biệt hoàn toàn hai quá trình: Đầu tiên dùng depthwise convolution áp dụng riêng biệt trên từng kênh, sau đó dùng pointwise convolution (1x1) để trộn các kênh lại với nhau.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Tích chập phân tách chiều sâu (SeparableConv) được thiết kế dựa trên một giả định quan trọng nào về dữ liệu hình ảnh?",
    options: [
      "Vị trí không gian độc lập với nhau, nhưng các kênh ảnh tương quan cao.",
      "Các vị trí không gian tương quan cao, nhưng các kênh có tính độc lập cao.",
      "Không có bất kỳ sự tương quan nào giữa không gian và các kênh màu ảnh.",
      "Tất cả các điểm ảnh trong ảnh đều có thể được tính trung bình gộp lại."
    ],
    correctAnswer: 1,
    explanation: "SeparableConv giả định rằng không gian ảnh (các điểm ảnh gần nhau) có tính tương quan cực lớn, trong khi các đặc trưng trên các kênh (channel-wise) thì mang tính độc lập và có thể xử lý tách rời ở giai đoạn đầu.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Mô hình nào sau đây thuộc thư viện Keras được xây dựng hoàn toàn dựa trên các khối Depthwise Separable Convolutions?",
    options: [
      "Mạng lưới ResNet50",
      "Mạng lưới VGG16",
      "Mạng lưới Xception",
      "Mạng lưới InceptionV3"
    ],
    correctAnswer: 2,
    explanation: "Xception (Extreme Inception) là kiến trúc được xây dựng hoàn toàn dựa trên SeparableConv2D, mang lại hiệu năng ấn tượng với tham số ít hơn.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "So sánh một lớp SeparableConv2D và một lớp Conv2D thông thường (cùng bộ lọc và đầu vào), SeparableConv2D có đặc điểm nào?",
    options: [
      "Nhiều tham số huấn luyện hơn gấp đôi và chạy chậm hơn.",
      "Rất ít tham số huấn luyện hơn nhưng vẫn giữ nguyên biểu diễn.",
      "Không có bất kỳ tham số huấn luyện nào do nó chỉ làm phép gộp.",
      "Sử dụng chung một số lượng tham số nhưng yêu cầu bộ nhớ RAM ít."
    ],
    correctAnswer: 1,
    explanation: "SeparableConv2D giảm mạnh số lượng tham số và số phép tính toán dấu phẩy động (flops) do nó tách biệt tích chập thay vì làm tích chập 3D đồng thời, giúp mô hình nhỏ và khó bị overfitting hơn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Mặc dù ít tham số hơn rất nhiều, tại sao SeparableConv2D thường không thể hiện tốc độ chạy nhanh hơn rõ rệt trên GPU so với Conv2D?",
    options: [
      "Vì SeparableConv2D bắt buộc phải sử dụng vi xử lý CPU thay vì GPU.",
      "Vì hệ sinh thái phần cứng GPU (cuDNN) đã tối ưu hóa quá mức cho Conv2D.",
      "Vì thư viện Keras chưa hỗ trợ chạy SeparableConv2D song song trên luồng.",
      "Vì SeparableConv2D đòi hỏi kích thước batch size siêu lớn để có thể chạy."
    ],
    correctAnswer: 1,
    explanation: "Các thuật toán Conv2D thông thường đã được tối ưu vi mô tới từng lệnh máy trong cuDNN của Nvidia suốt nhiều năm. Các thuật toán mới như SeparableConv chưa có mức độ tối ưu phần cứng tương đương.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Tại sao không nên áp dụng lớp SeparableConv2D ngay cho lớp nhận ảnh gốc (RGB) đầu tiên trong mạng lưới?",
    options: [
      "Vì ảnh RGB không thể sử dụng kỹ thuật zero-padding để duy trì kích thước.",
      "Vì ba kênh màu Đỏ, Lục, Lam có tính tương quan cực kỳ cao trong thực tế.",
      "Vì lớp SeparableConv2D chỉ chấp nhận hình ảnh trắng đen (grayscale 1 kênh).",
      "Vì ảnh gốc chưa được chuẩn hóa lô (batch normalization) để đưa vào mạng."
    ],
    correctAnswer: 1,
    explanation: "Giả định của SeparableConv là các kênh độc lập với nhau. Tuy nhiên, ở ảnh thực tế, 3 kênh R, G, B có sự tương quan cực lớn, do đó lớp đầu tiên nên dùng Conv2D thông thường.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Kiến trúc nào sau đây ban đầu được thiết kế để xử lý văn bản, nhưng gần đây đã trở thành đối thủ đáng gờm của ConvNet trong xử lý hình ảnh?",
    options: [
      "Mạng Recurrent Neural Network (RNN)",
      "Mạng Bộ nhớ ngắn hạn dài (LSTM)",
      "Mạng Vision Transformers (ViTs)",
      "Mạng Multilayer Perceptron (MLP)"
    ],
    correctAnswer: 2,
    explanation: "Transformer ban đầu được thiết kế cho chuỗi ngôn ngữ. Vision Transformers (ViTs) chia ảnh thành các bản vá (patches) như chuỗi 1D và xử lý chúng, là đối thủ lớn hiện tại của ConvNets.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Khi làm việc với các bộ dữ liệu nhỏ, tại sao ConvNets thường vượt trội hơn Vision Transformers (ViTs)?",
    options: [
      "ConvNets xử lý hình ảnh dựa trên giả định không gian và cấu trúc cục bộ 2D.",
      "Vision Transformers không thể chạy được nếu thiếu các GPU hàng đầu.",
      "ConvNets có khả năng tạo ra dữ liệu tự động bên trong cấu trúc mạng.",
      "Vision Transformers tự động loại bỏ màu sắc của ảnh khiến dữ liệu thiếu."
    ],
    correctAnswer: 0,
    explanation: "ConvNets tích hợp sẵn ưu tiên không gian 2D (spatial prior), giúp nó trích xuất đặc trưng với lượng dữ liệu rất nhỏ. ViTs bắt đầu với ít giả định hơn nên cần lượng dữ liệu khổng lồ để tự học cấu trúc.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong một kiến trúc mini-Xception, thứ tự của một khối ConvBlock tiêu chuẩn là gì?",
    options: [
      "Conv2D -> MaxPooling -> Activation -> BatchNormalization.",
      "BatchNormalization -> Activation -> SeparableConv2D.",
      "Activation -> SeparableConv2D -> BatchNormalization.",
      "MaxPooling -> SeparableConv2D -> BatchNormalization -> Activation."
    ],
    correctAnswer: 1,
    explanation: "Như trong code mẫu của tác giả Chollet ở chương 9, cấu trúc khối gồm: BatchNormalization -> Activation -> SeparableConv2D (do đầu ra vòng lặp trước đã là đầu vào convolution).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Phép kết nối dư (Residual connection) khắc phục hiện tượng mất mát đạo hàm (vanishing gradient) chủ yếu nhờ vào tính chất gì?",
    options: [
      "Nó đóng vai trò như một bộ khuếch đại tín hiệu nhiễu cực đại.",
      "Nó loại bỏ hẳn hàm kích hoạt phi tuyến trên suốt quá trình lan truyền.",
      "Nó cung cấp một đường cao tốc thông tin phi phá hủy (noiseless shortcut).",
      "Nó đưa toàn bộ trọng số của các lớp về mức 0 một cách tuần hoàn liên tục."
    ],
    correctAnswer: 2,
    explanation: "Residual shortcut đưa trực tiếp đầu vào lên phía trước mà không bị biến đổi, cho phép tín hiệu gradient lỗi truyền ngược (backpropagate) mượt mà mà không bị nhân với các giá trị nhỏ liên tục.",
    difficulty: "Dễ"
  },
  {
    type: "fill",
    question: "Trong kiến trúc học sâu, {1} (residual connection) là kỹ thuật thêm trực tiếp đầu vào của một khối tính toán vào thẳng đầu ra của khối đó, tạo lối tắt truyền dẫn thông tin.",
    blanks: [
      { id: 1, text: "kết nối dư", answer: "kết nối dư" }
    ],
    explanation: "Kết nối dư (Residual connection) là đột phá thiết kế giúp xây dựng các mạng rất sâu (ResNet) mà không bị vanishing gradients.",
    difficulty: "Dễ"
  },
  {
    type: "fill",
    question: "Thay vì thực hiện phép chập 3D đắt đỏ, kỹ thuật {1} tách biệt ra thành chập không gian trên từng kênh, sau đó nối lại bằng chập 1x1.",
    blanks: [
      { id: 1, text: "tích chập phân tách chiều sâu", answer: "tích chập phân tách chiều sâu|separable convolution" }
    ],
    explanation: "Tích chập phân tách chiều sâu (Depthwise Separable Convolution) tách riêng việc học đặc trưng không gian và đặc trưng kênh màu.",
    difficulty: "Trung bình"
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự một chu trình của lớp SeparableConv2D (Tích chập có thể phân tách theo chiều sâu):",
    steps: [
      "Nhận đầu vào là tensor có nhiều kênh (Channels).",
      "Thực hiện tích chập không gian (Depthwise) độc lập trên TỪNG kênh riêng biệt.",
      "Thực hiện phép tích chập theo điểm (Pointwise 1x1) để trộn đặc trưng các kênh.",
      "Tạo ra bản đồ đặc trưng đầu ra với số kênh (channels) mới mong muốn."
    ],
    explanation: "Quy trình của SeparableConv: Đầu vào -> Xử lý không gian độc lập (Depthwise) -> Trộn kênh lại (Pointwise) -> Đầu ra mới.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Ghép nối các khái niệm về kiến trúc mô hình học sâu với vai trò của chúng:",
    pairs: [
      { left: "Residual Connection", right: "Tạo lối tắt thông tin giúp tránh triệt tiêu đạo hàm" },
      { left: "Batch Normalization", right: "Chuẩn hóa động phân phối dữ liệu trong quá trình huấn luyện" },
      { left: "SeparableConv2D", right: "Giảm mạnh lượng tham số bằng cách tách biệt không gian và kênh" },
      { left: "Vision Transformers", right: "Xử lý ảnh dưới dạng các mảng dữ liệu chuỗi rời rạc" }
    ],
    explanation: "Đây là 4 khái niệm quan trọng nhất được nhắc đến trong Chương 9.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Ghép nối chức năng kỹ thuật trong việc thao tác với phần dư (Residual):",
    pairs: [
      { left: "Sử dụng Conv2D với kernel 1x1", right: "Đồng bộ số lượng kênh khi phần dư khác số kênh đầu ra khối" },
      { left: "Sử dụng Conv2D với strides=2", right: "Đồng bộ kích thước không gian khi khối có MaxPooling" },
      { left: "Sử dụng padding='same'", right: "Giữ nguyên bề rộng/chiều cao giữa tensor đầu vào và đầu ra" },
      { left: "Lệnh add([x, residual])", right: "Kết hợp khối lượng thông tin tính toán mới và lối tắt gốc" }
    ],
    explanation: "Để thực hiện phép cộng `add` trong Residual Block, kích thước không gian (strides, padding) và kích thước kênh (1x1 Conv) phải hoàn toàn khớp nhau.",
    difficulty: "Khó"
  }
];

export default questions;
