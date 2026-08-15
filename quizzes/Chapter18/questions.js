const questions = [
  {
    type: "mcq",
    question: "Sự khác biệt cơ bản giữa 'tham số' (parameters) và 'siêu tham số' (hyperparameters) của một mô hình là gì?",
    options: [
      "Tham số là các trọng số được học tự động thông qua quá trình lan truyền ngược, còn siêu tham số là các quyết định thiết kế kiến trúc do con người hoặc thuật toán đặt ra trước khi huấn luyện.",
      "Tham số chỉ dùng cho các mạng nơ-ron nhỏ dưới 3 lớp, còn siêu tham số bắt buộc dùng cho các mạng lưới thần kinh sâu (Deep Neural Networks) khổng lồ.",
      "Tham số được lưu trữ trên bộ nhớ RAM của máy tính, còn siêu tham số phải được lưu trực tiếp trên bộ nhớ đệm tốc độ cao của card đồ họa (VRAM).",
      "Tham số chỉ thay đổi khi có sự can thiệp của người dùng, trong khi siêu tham số tự động thay đổi ở mỗi bước sóng của trình tối ưu hóa Adam."
    ],
    answer: 0,
    explanation: "Tham số (ví dụ: trọng số w, b) được cập nhật bằng Gradient Descent. Siêu tham số (ví dụ: số layer, số units, learning rate) phải được chọn trước quá trình huấn luyện."
  },
  {
    type: "mcq",
    question: "Tại sao không thể sử dụng phương pháp giảm độ dốc (Gradient Descent) thông thường để tối ưu hóa trực tiếp không gian siêu tham số?",
    options: [
      "Vì không gian siêu tham số thường rời rạc (ví dụ: số lớp mạng) và không khả vi, không thể tính được đạo hàm để điều chỉnh bằng gradient.",
      "Vì Keras và TensorFlow hiện tại chưa hỗ trợ API nào cho phép tính toán đạo hàm trên các tham số có kiểu dữ liệu là số nguyên (Integer).",
      "Vì gradient descent luôn yêu cầu dữ liệu phải tuân theo phân phối chuẩn, trong khi các siêu tham số có tính chất phân bố hoàn toàn ngẫu nhiên.",
      "Vì việc tính đạo hàm trên siêu tham số sẽ ngay lập tức làm tràn bộ nhớ (Out of Memory) trên bất kỳ chiếc GPU thương mại nào hiện nay."
    ],
    answer: 0,
    explanation: "Hầu hết siêu tham số (như chọn hàm kích hoạt 'relu' vs 'tanh', số lớp mạng) là những lựa chọn rời rạc, làm cho hàm mục tiêu không trơn tru và không thể lấy đạo hàm."
  },
  {
    type: "mcq",
    question: "Trong công cụ KerasTuner, lớp `BayesianOptimization` hoạt động dựa trên nguyên lý nào để tìm siêu tham số tốt nhất?",
    options: [
      "Nó cố gắng đưa ra dự đoán thông minh về bộ siêu tham số mới có khả năng hoạt động tốt nhất dựa trên kết quả của các lựa chọn đã thử trước đó.",
      "Nó sinh ngẫu nhiên hàng triệu cấu hình khác nhau cùng lúc và chọn cấu hình cuối cùng bằng cách tung đồng xu cho đến khi có một mô hình tốt.",
      "Nó duyệt qua tất cả mọi sự kết hợp có thể (Grid Search) theo thứ tự từ điển cho đến khi tìm thấy cấu hình đạt độ chính xác chính xác 100%.",
      "Nó tải mã nguồn của các mô hình chiến thắng trên Kaggle về, sau đó trực tiếp áp dụng toàn bộ kiến trúc đó vào tập dữ liệu của bạn mà không cần chạy."
    ],
    answer: 0,
    explanation: "Thay vì tìm kiếm ngẫu nhiên (Random Search), Tối ưu hóa Bayes sử dụng xác suất và lịch sử các lần chạy trước (trials) để đoán xem vùng không gian nào tiềm năng nhất."
  },
  {
    type: "mcq",
    question: "Khi thực hiện tối ưu hóa siêu tham số tự động, rủi ro lớn nhất liên quan đến tập dữ liệu xác thực (Validation set) là gì?",
    options: [
      "Mô hình sẽ nhanh chóng bị trang bị quá mức (overfit) đối với tập xác thực, vì bạn đang liên tục chọn các siêu tham số làm cho điểm số trên tập này tăng lên.",
      "Tập xác thực sẽ bị thu nhỏ kích thước liên tục sau mỗi lần thử, khiến mô hình không còn đủ dữ liệu để đánh giá tính tổng quát một cách chính xác.",
      "KerasTuner sẽ tự động trộn lẫn (shuffle) tập huấn luyện và tập xác thực vào với nhau, phá vỡ tính toàn vẹn của dữ liệu chuỗi thời gian.",
      "Điểm số của tập xác thực sẽ ngày càng thấp đi do hiện tượng nổ gradient, làm cho KerasTuner hiểu lầm và đưa ra siêu tham số tồi."
    ],
    answer: 0,
    explanation: "Hiện tượng 'Validation-set overfitting' rất dễ xảy ra khi chạy Hyperparameter Tuning. Vì vậy, ta luôn cần một tập Test độc lập để đánh giá mô hình cuối cùng."
  },
  {
    type: "mcq",
    question: "Yếu tố cốt lõi nào quyết định sự thành công khi áp dụng kỹ thuật Tổ hợp mô hình (Model Ensembling)?",
    options: [
      "Sự đa dạng (Diversity) của các mô hình: Các mô hình thành phần nên càng tốt càng tốt nhưng phải khác biệt nhau về cách tiếp cận hoặc kiến trúc.",
      "Số lượng mô hình: Cần phải có ít nhất 1000 mô hình Neural Network nhỏ kết hợp lại thì quá trình Ensembling mới cho ra kết quả tốt hơn mô hình đơn.",
      "Độ phức tạp: Mọi mô hình trong tổ hợp đều bắt buộc phải là mạng nơ-ron học sâu nhiều lớp, không được sử dụng các thuật toán học máy cổ điển.",
      "Sự giống nhau (Similarity): Các mô hình phải có cùng chung một kiến trúc và chỉ được khác nhau duy nhất ở hạt giống khởi tạo (random seed) ban đầu."
    ],
    answer: 0,
    explanation: "Giống như ví dụ 'Thầy bói xem voi', sự đa dạng (biases in different ways) giúp các lỗi sai của từng mô hình tự triệt tiêu lẫn nhau khi tính trung bình."
  },
  {
    type: "mcq",
    question: "Chiến lược 'Song song dữ liệu' (Data parallelism) hoạt động như thế nào trong đào tạo đa GPU?",
    options: [
      "Một bản sao toàn vẹn của mô hình được đặt trên mỗi GPU; lô dữ liệu (batch) được chia nhỏ thành các sub-batches, xử lý song song rồi tính trung bình gradient.",
      "Mô hình được cắt làm nhiều mảnh (ví dụ nửa đầu vào GPU 1, nửa sau GPU 2) và mỗi GPU chỉ xử lý một phần của tính toán, nhưng dùng chung một lô dữ liệu.",
      "Tập dữ liệu được giữ cố định trên ổ cứng, các GPU thay phiên nhau đọc từng file dữ liệu một và cập nhật mô hình tuần tự để tránh xung đột.",
      "Các GPU sẽ huấn luyện song song nhiều kiến trúc mô hình khác biệt nhau hoàn toàn, sau đó tự động Ensembling chúng lại vào cuối quá trình."
    ],
    answer: 0,
    explanation: "Data parallelism chia nhỏ Data (divide and conquer). Mỗi GPU tính forward/backward trên sub-batch của mình, sau đó Gradients được gộp lại (All-Reduce) để cập nhật trọng số đồng loạt."
  },
  {
    type: "mcq",
    question: "Trong trường hợp nào ta bắt buộc phải dùng chiến lược 'Song song mô hình' (Model parallelism) thay vì 'Song song dữ liệu'?",
    options: [
      "Khi toàn bộ kiến trúc mô hình (ví dụ một LLM khổng lồ) quá lớn để vừa vặn trong bộ nhớ VRAM của bất kỳ một GPU đơn lẻ nào.",
      "Khi tập dữ liệu huấn luyện quá lớn (ví dụ hàng Terabyte), ổ cứng không thể đọc kịp để cấp cho tất cả các GPU cùng lúc.",
      "Khi mô hình cần phải dự đoán kết quả ngay lập tức trên các thiết bị biên (Edge Devices) như điện thoại di động hoặc Raspberry Pi.",
      "Khi ta muốn tận dụng GPU để vẽ giao diện đồ họa (GUI) song song với việc huấn luyện mô hình mạng nơ-ron học sâu."
    ],
    answer: 0,
    explanation: "Model parallelism cắt nhỏ mô hình (Sharding) và chia các lớp/biến lên nhiều GPU khác nhau để giải quyết vấn đề vượt quá giới hạn bộ nhớ (Out of Memory)."
  },
  {
    type: "mcq",
    question: "Trong môi trường nhiều thiết bị, lớp `LayoutMap` của Keras thực hiện nhiệm vụ gì?",
    options: [
      "Nó đóng vai trò như một từ điển ánh xạ các đường dẫn biến (variable path) với thông tin chỉ định xem biến đó nên được nhân bản hay chia nhỏ dọc theo trục lưới thiết bị.",
      "Nó trực tiếp tải tất cả các hình ảnh từ thư mục cục bộ và phân phối đều cho các nhân CPU nhằm tăng tốc quá trình tiền xử lý dữ liệu.",
      "Nó tự động thiết kế kiến trúc các tầng ẩn trong một mạng nơ-ron thông qua quá trình dò tìm để đảm bảo mô hình không bị quá khớp.",
      "Nó vẽ một bản đồ đồ họa trực quan về vị trí vật lý của từng con chip TPU trên máy chủ Google Cloud để người lập trình theo dõi nhiệt độ."
    ],
    answer: 0,
    explanation: "`LayoutMap` cho phép Keras biết một tensor trọng số cụ thể nên được sao chép (replicated) toàn vẹn hay bị cắt nhỏ (sharded) theo các chiều của lưới thiết bị (Device Mesh)."
  },
  {
    type: "mcq",
    question: "Đào tạo có độ chính xác hỗn hợp (Mixed-precision training) kết hợp `float16` và `float32` như thế nào?",
    options: [
      "Quá trình tính toán tiến (forward pass) phần lớn chạy bằng `float16` cho tốc độ cao, nhưng trọng số và cập nhật gradient được lưu ở `float32` để giữ độ ổn định.",
      "Các lớp tích chập (Conv2D) luôn chạy ở `float32` để không mất chi tiết ảnh, trong khi các lớp dầy (Dense) chạy ở `float16` để giảm số lượng tham số.",
      "Ở mỗi epoch chẵn, toàn bộ mô hình chạy bằng `float16`, và ở mỗi epoch lẻ, mô hình tự động chuyển sang `float32` để bù đắp sự mất mát thông tin.",
      "Kỹ thuật này dùng `float16` để tính giá trị lỗi (Loss) cuối cùng, nhưng lại dùng `float32` để lấy mẫu ngẫu nhiên các lô dữ liệu (Batches)."
    ],
    answer: 0,
    explanation: "Phép nhân ma trận trong forward pass không cần quá chính xác, có thể dùng `float16`. Nhưng phép cập nhật trọng số (Gradient update) rất nhỏ, đòi hỏi phải dùng `float32` để không bị làm tròn về 0."
  },
  {
    type: "mcq",
    question: "Khi huấn luyện Mixed-precision, tại sao ta thường phải dùng `LossScaleOptimizer` hoặc nhân Loss với một hằng số lớn (loss_scale_factor)?",
    options: [
      "Vì giá trị gradient thường rất nhỏ, nếu ở dạng float16 chúng dễ bị tràn dưới (underflow) và làm tròn xuống 0, việc nhân lên giúp giữ được tín hiệu gradient.",
      "Vì float16 làm mất đi dấu âm của các con số, nên hàm loss bắt buộc phải được nhân với 100 để đẩy tất cả các giá trị trở về vùng dương lớn hơn 0.",
      "Vì quá trình đào tạo sẽ diễn ra quá nhanh, việc tăng hệ số Loss giúp mô hình hãm phanh lại và học tập từ từ để tránh trượt qua điểm cực tiểu.",
      "Vì hàm kích hoạt Softmax luôn trả về kết quả bằng 1, nên việc chia tỷ lệ loss là cách duy nhất để Keras nhận biết được khi nào mô hình đang dự đoán sai."
    ],
    answer: 0,
    explanation: "Loss scaling là kỹ thuật nhân hàm Loss lên một lượng (VD: 1024), kéo theo Gradients cũng tăng 1024 lần, đưa các gradient rất nhỏ (1e-6) vượt qua giới hạn độ phân giải của float16 để không bị biến thành 0."
  },
  {
    type: "mcq",
    question: "Kỹ thuật 'Lượng tử hóa int8' (int8 Quantization) mang lại lợi ích gì lớn nhất cho mô hình học sâu?",
    options: [
      "Giảm dung lượng mô hình và tăng tốc độ suy luận đáng kể bằng cách chuyển trọng số từ float32 sang int8, mà ít làm suy giảm độ chính xác dự đoán.",
      "Giúp mô hình có khả năng học tập liên tục trên tập dữ liệu mới (Online Learning) mà không cần phải thực hiện quá trình lan truyền ngược.",
      "Cho phép mô hình Keras có thể chạy trực tiếp trên các trình duyệt web đời cũ mà không cần phải cài đặt thêm WebGL hay WebGPU.",
      "Thay thế hoàn toàn bộ dữ liệu huấn luyện bằng các số nguyên dương từ 0 đến 255 để tránh vi phạm bản quyền dữ liệu hình ảnh."
    ],
    answer: 0,
    explanation: "Int8 Quantization chia tỷ lệ (scale) các số thực float32 vào khoảng số nguyên [-127, 127], giúp quá trình nhân ma trận (matmul) chạy cực nhanh và tiết kiệm RAM lúc Inference."
  },
  {
    type: "mcq",
    question: "Trong Colab hoặc Google Cloud, thủ thuật `steps_per_execution` có ý nghĩa gì khi huấn luyện trên TPU?",
    options: [
      "Cho phép TPU thực thi liền mạch nhiều bước đào tạo (batches) trước khi gửi kết quả về máy ảo (CPU), giúp giảm nghẽn cổ chai truyền tải và tăng hiệu suất sử dụng.",
      "Bắt buộc TPU phải tạm dừng nghỉ ngơi sau mỗi 100 bước để làm mát phần cứng, tránh làm cháy chip trong quá trình huấn luyện các mô hình quá lớn.",
      "Chia nhỏ một lô dữ liệu (batch) khổng lồ thành các mảnh nhỏ hơn để một lõi TPU duy nhất có thể nhai nuốt từ từ mà không bị cạn kiệt bộ nhớ.",
      "Báo hiệu cho Keras biết cần phải gọi hàm sao lưu trọng số (ModelCheckpoint) bao nhiêu lần trong một kỷ nguyên (epoch)."
    ],
    answer: 0,
    explanation: "Quá trình giao tiếp giữa máy ảo chủ (CPU) và TPU tốn một lượng chi phí trễ. Kỹ thuật 'step fusing' gộp n bước chạy thành 1 lần gọi phần cứng, tận dụng tối đa sức mạnh tính toán kinh khủng của TPU."
  },
  {
    type: "fill",
    question: "Cấu hình kiến trúc do kỹ sư thiết lập trước khi huấn luyện (như số layer, learning rate) được gọi là ______________ (hyperparameters).",
    options: ["siêu tham số", "siêu tham số", "hyperparameters"],
    answer: "siêu tham số",
    explanation: "Siêu tham số (Hyperparameters) phân biệt với Tham số (Parameters - các trọng số w, b mà mô hình tự học)."
  },
  {
    type: "fill",
    question: "Kỹ thuật tổng hợp dự đoán của nhiều mô hình khác nhau để tạo ra một kết quả tốt hơn được gọi là ______________ (Model Ensembling).",
    options: ["tổ hợp mô hình", "tổ hợp", "ensembling"],
    answer: "tổ hợp mô hình",
    explanation: "Tổ hợp mô hình (Ensembling) là kỹ thuật quan trọng giúp cải thiện độ chính xác và đạt top trong các cuộc thi Kaggle."
  },
  {
    type: "fill",
    question: "Chiến lược chia lô dữ liệu thành các phần nhỏ và xử lý đồng thời trên nhiều GPU rồi tính trung bình đạo hàm gọi là song song ______________ (Data parallelism).",
    options: ["dữ liệu", "data"],
    answer: "dữ liệu",
    explanation: "Data parallelism phân chia dữ liệu cho các bản sao nguyên vẹn của mô hình (replicas)."
  },
  {
    type: "fill",
    question: "Kiểu dữ liệu float16 và bfloat16 đều sử dụng 16 bit, tuy nhiên `bfloat16` dành nhiều bit hơn cho phần ______________ (exponent) để mở rộng dải giá trị biểu diễn.",
    options: ["số mũ", "mũ", "exponent"],
    answer: "số mũ",
    explanation: "bfloat16 hi sinh số bit của mantissa (độ phân giải) để có 8 bit cho số mũ (tương đương với float32), giúp biểu diễn số rất lớn/rất nhỏ mà không tràn."
  },
  {
    type: "fill",
    question: "Để tăng tốc quá trình suy luận (Inference), người ta thường áp dụng kỹ thuật chuyển trọng số từ float32 về số nguyên 8-bit, gọi là ______________ int8 (Quantization).",
    options: ["lượng tử hóa", "Lượng tử hóa", "quantization"],
    answer: "lượng tử hóa",
    explanation: "Lượng tử hóa (Quantization) ánh xạ khoảng giá trị thực sang khoảng số nguyên [-127, 127], tính toán rất nhanh trên phần cứng."
  },
  {
    type: "matching",
    question: "Ghép nối các kỹ thuật tối ưu hóa siêu tham số (Hyperparameter Tuning):",
    options: [
      "Random Search",
      "Grid Search",
      "Bayesian Optimization"
    ],
    answer: [
      "Thử nghiệm ngẫu nhiên các tổ hợp siêu tham số trong không gian tìm kiếm.",
      "Duyệt cạn, thử mọi tổ hợp siêu tham số có thể có một cách có hệ thống.",
      "Dựa vào kết quả các lần thử trước để đoán xem vùng siêu tham số nào tốt nhất để thử tiếp."
    ],
    explanation: "Bayesian Optimization hiệu quả nhất vì nó 'học' từ lịch sử để không lãng phí thời gian vào các vùng không gian kém chất lượng."
  },
  {
    type: "matching",
    question: "Ghép nối các khái niệm trong phân tán mô hình (Distributed Training):",
    options: [
      "Data parallelism (Song song dữ liệu)",
      "Model parallelism (Song song mô hình)",
      "Device Mesh (Lưới thiết bị)"
    ],
    answer: [
      "Sao chép mô hình lên mọi GPU, chia nhỏ lô dữ liệu cho mỗi GPU tính toán.",
      "Cắt nhỏ các lớp của một mô hình siêu lớn (như LLM) ra và đặt lên nhiều GPU.",
      "Kiến trúc logic mô tả cách các phần cứng (như 8 GPU) được tổ chức thành các trục tính toán."
    ],
    explanation: "Hiểu được khác biệt giữa Data/Model parallelism là nền tảng để đào tạo các mô hình Deep Learning cỡ lớn."
  },
  {
    type: "matching",
    question: "So sánh độ chính xác của các kiểu dữ liệu dấu phẩy động (Floating point):",
    options: [
      "float32 (Độ chính xác đơn)",
      "float16 (Độ chính xác một nửa)",
      "bfloat16 (Brain float)"
    ],
    answer: [
      "Sử dụng 32 bit, độ chính xác cao nhất trong nhóm, dùng lưu trữ cập nhật trọng số an toàn.",
      "Sử dụng 16 bit (5 bit mũ, 10 bit dải), tính toán nhanh, nhưng dải giá trị nhỏ, dễ tràn underflow.",
      "Sử dụng 16 bit (8 bit mũ, 7 bit dải), dải giá trị rất rộng (như float32) nhưng độ phân giải thấp hơn, tối ưu trên TPU."
    ],
    explanation: "Trong Deep Learning hiện đại, float32 đang dần nhường chỗ cho float16/bfloat16 để tăng tốc độ xử lý phần cứng."
  },
  {
    type: "matching",
    question: "Ghép nối các kỹ thuật cải thiện huấn luyện thực tế:",
    options: [
      "Loss Scaling (Mở rộng tỷ lệ tổn thất)",
      "Model Ensembling (Tổ hợp mô hình)",
      "steps_per_execution (Gộp bước)"
    ],
    answer: [
      "Nhân Loss với số lớn để Gradients không bị làm tròn về 0 khi dùng float16.",
      "Trung bình trọng số dự đoán của nhiều mô hình đa dạng để khử sai số cục bộ.",
      "Yêu cầu TPU tính toán nhiều batches cùng lúc trước khi trả kết quả về CPU để tránh nghẽn cổ chai."
    ],
    explanation: "Mỗi kỹ thuật giải quyết một 'nỗi đau' riêng: thất thoát gradient, độ chính xác, và tốc độ truyền tải TPU."
  },
  {
    type: "matching",
    question: "Ghép nối các khái niệm trong Kỹ thuật Lượng tử hóa int8 (Quantization):",
    options: [
      "abs_max_quantize",
      "Scale factor (Hệ số chia tỷ lệ)",
      "Un-scale (Hoàn nguyên tỷ lệ)"
    ],
    answer: [
      "Phương pháp kẹp (clip) và chuyển đổi giá trị float của tensor về khoảng [-127, 127].",
      "Đại lượng được tính bằng cách lấy 127 chia cho giá trị tuyệt đối lớn nhất của tensor đầu vào.",
      "Chia kết quả int8 thu được cho hệ số tỷ lệ để đưa kết quả cuối cùng trở về dạng float32."
    ],
    explanation: "Quantization int8 lợi dụng tính chất tuyến tính của phép nhân ma trận để tính toán ở vùng số nguyên cực nhanh, sau đó đổi lại thành số thực."
  },
  {
    type: "sorting",
    question: "Sắp xếp các bước thực hiện tự động tìm kiếm siêu tham số bằng KerasTuner:",
    options: [
      "Xây dựng hàm `build_model(hp)` định nghĩa không gian tìm kiếm (ví dụ: hp.Int, hp.Choice).",
      "Khởi tạo một Tuner (ví dụ: BayesianOptimization) và chỉ định hàm mục tiêu (val_accuracy).",
      "Gọi hàm `tuner.search()` truyền vào tập huấn luyện và tập xác thực.",
      "Truy vấn cấu hình tốt nhất bằng `tuner.get_best_hyperparameters(top_n=1)`.",
      "Huấn luyện lại mô hình với cấu hình tốt nhất trên toàn bộ tập dữ liệu (bao gồm cả dữ liệu xác thực)."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Quy trình chuẩn: Định nghĩa không gian -> Khởi tạo Tuner -> Chạy dò tìm -> Lấy kết quả -> Train chốt sổ."
  },
  {
    type: "sorting",
    question: "Sắp xếp quá trình huấn luyện độ chính xác hỗn hợp (Mixed-precision) trong một bước lan truyền:",
    options: [
      "Thiết lập `keras.config.set_dtype_policy('mixed_float16')` trước khi khởi tạo mô hình.",
      "Mô hình nhận đầu vào, tự động ép kiểu (cast) và tính toán chuyển tiếp (forward pass) chủ yếu bằng `float16`.",
      "Tính toán Hàm mất mát (Loss) và nhân với hệ số Loss Scale (nếu dùng LossScaleOptimizer).",
      "Truyền ngược tính toán đạo hàm (Gradients), giữ nguyên định dạng `float16`.",
      "Chuyển đổi Gradients về `float32` và cập nhật các trọng số mô hình (cũng đang được lưu ở `float32`)."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Trọng số chính (master weights) luôn ở dạng float32. Chỉ lúc chạy tính toán mới hạ xuống float16 để GPU tăng tốc nhân ma trận."
  },
  {
    type: "sorting",
    question: "Sắp xếp vòng đời lý tưởng của 'Vòng lặp tiến bộ' (Loop of Progress) trong Machine Learning:",
    options: [
      "Lên ý tưởng thiết kế kiến trúc hoặc cách giải quyết vấn đề.",
      "Lập trình mã nguồn Keras để thiết lập bài thí nghiệm nhanh chóng.",
      "Sử dụng hạ tầng huấn luyện tốc độ cao (Multi-GPU/TPU) để chạy mô hình.",
      "Phân tích kết quả, đọc các thông số số liệu (Metrics) trả về.",
      "Điều chỉnh ý tưởng và bước vào vòng lặp mới với chất lượng mô hình cao hơn."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Sự khác biệt giữa một Data Scientist bình thường và chuyên gia là tốc độ xoay vòng Vòng lặp tiến bộ này nhờ nắm bắt công cụ mạnh."
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự các bước trong phép tính Lượng tử hóa int8 (int8 Quantization) cho phép nhân `matmul(X, Kernel)`:",
    options: [
      "Xác định hệ số tỷ lệ (scale) cho X và Kernel dựa trên giá trị tuyệt đối lớn nhất (abs-max).",
      "Nhân X và Kernel với hệ số tỷ lệ tương ứng, sau đó làm tròn (round) và kẹp (clip) trong khoảng [-127, 127].",
      "Ép kiểu (Cast) X và Kernel thành số nguyên `int8`.",
      "Thực hiện phép nhân ma trận `matmul` bằng phần cứng xử lý số nguyên `int8` tốc độ cao.",
      "Ép kiểu kết quả ngược lại `float32` và chia cho tích của hai hệ số tỷ lệ ban đầu."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Đó là thuật toán abs-max lượng tử hóa đối xứng (symmetric quantization) rất cơ bản nhưng vô cùng hiệu quả."
  },
  {
    type: "sorting",
    question: "Sắp xếp quy trình thiết lập phân tán tính toán Mô hình (Model Parallelism) với Keras:",
    options: [
      "Tạo một `DeviceMesh` định nghĩa các trục tính toán (ví dụ: trục 'data', trục 'model').",
      "Kiểm tra đường dẫn biến (variable path) của mô hình, ví dụ: 'sequential/dense/kernel'.",
      "Thiết lập `LayoutMap` chỉ định biến nào sẽ được chia nhỏ dọc theo trục 'model'.",
      "Áp dụng cấu hình bằng `keras.distribution.set_distribution(ModelParallel(...))`.",
      "Xây dựng mô hình và gọi `fit()`, Keras sẽ tự động xử lý chia tách phía dưới."
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: "Nhờ JAX và Keras 3, việc huấn luyện phân tán LLM khổng lồ giờ đây chỉ cần vài dòng khai báo cấu hình LayoutMap."
  },
  {
    type: "mcq",
    question: "KerasTuner cung cấp các 'siêu mô hình' (hypermodels) tạo sẵn (như HyperXception, HyperResNet) nhằm mục đích gì?",
    options: [
      "Để cung cấp các không gian tìm kiếm phù hợp sẵn cho các bài toán phổ biến như phân loại hình ảnh, giúp tiết kiệm thời gian thiết kế kiến trúc.",
      "Để loại bỏ hoàn toàn quá trình huấn luyện bằng cách tải trực tiếp trọng số tốt nhất từ Google Cloud xuống máy tính của bạn.",
      "Để tăng tốc độ huấn luyện trên GPU bằng cách tắt tính toán đạo hàm của các lớp này trong quá trình dò tìm siêu tham số.",
      "Để tạo ra các hình ảnh giả nhằm đánh lừa bộ phân loại và tìm ra lỗ hổng bảo mật của mô hình mạng nơ-ron."
    ],
    answer: 0,
    explanation: "Không gian tìm kiếm phát triển theo tổ hợp và rất khó thiết kế. KerasTuner cung cấp các khuôn mẫu đã được tối ưu cho bài toán phổ biến để ta có thể dùng ngay."
  },
  {
    type: "mcq",
    question: "Tại sao không nên sử dụng tập dữ liệu Kiểm tra (Test set) làm dữ liệu Xác thực (Validation set) trong KerasTuner?",
    options: [
      "Vì KerasTuner sử dụng tín hiệu từ tập xác thực để chọn siêu tham số, điều này sẽ khiến mô hình bị quá khớp vào tập kiểm tra, làm mất đi tính đánh giá độc lập.",
      "Vì tập kiểm tra luôn có kích thước nhỏ hơn tập xác thực, khiến KerasTuner không đủ số lượng mẫu để chạy qua tất cả các lớp của mạng nơ-ron.",
      "Vì tập kiểm tra không chứa nhãn (labels), do đó trình tối ưu hóa không thể tính toán được giá trị loss để điều hướng thuật toán Bayesian Optimization.",
      "Vì việc sử dụng tập kiểm tra ở giai đoạn này sẽ vi phạm các điều khoản bảo mật dữ liệu của nền tảng Kaggle."
    ],
    answer: 0,
    explanation: "Nếu dùng Test set để dò hyperparameter, bạn đã vô tình 'rò rỉ' thông tin của Test set vào thiết kế mô hình, làm cho kết quả đánh giá cuối cùng không còn đáng tin cậy."
  },
  {
    type: "mcq",
    question: "Một trong những kỹ thuật hiệu quả để tăng tốc xử lý dữ liệu với API `tf.data` là gì?",
    options: [
      "Sử dụng `dataset.prefetch(tf.data.AUTOTUNE)` để nạp trước lô dữ liệu tiếp theo vào bộ nhớ trong khi GPU/TPU đang huấn luyện lô hiện tại.",
      "Lưu trữ dữ liệu dưới định dạng CSV thay vì TFRecord vì Pandas đọc file CSV nhanh hơn nhiều so với mã hóa nhị phân của TensorFlow.",
      "Tắt hoàn toàn tính năng xáo trộn dữ liệu (shuffle) để ổ cứng không phải nhảy (seek) ngẫu nhiên, giúp luồng dữ liệu mượt mà hơn.",
      "Truyền trực tiếp từng bức ảnh một dưới dạng mảng NumPy thay vì đóng gói thành lô (batch) để GPU xử lý ảnh nhanh hơn."
    ],
    answer: 0,
    explanation: "Prefetch (Tìm nạp trước) giúp tách biệt I/O và tính toán. Khi thiết bị đang bận train, CPU đã chuẩn bị sẵn batch tiếp theo."
  },
  {
    type: "mcq",
    question: "Điều gì KHÔNG ĐÚNG khi nói về đào tạo độ chính xác hỗn hợp (Mixed Precision) trong Keras?",
    options: [
      "Hàm kích hoạt `Softmax` và hàm mất mát `Crossentropy` luôn bắt buộc phải tính toán bằng `float16` để đảm bảo tốc độ cao nhất.",
      "Quá trình chuyển tiếp (forward pass) chủ yếu sử dụng `float16` để tăng tốc độ nhân ma trận.",
      "Các biến (variables) lưu trữ trọng số mô hình thường giữ ở định dạng `float32` để đảm bảo nhận cập nhật ổn định.",
      "Trình tối ưu hóa sẽ chia tỷ lệ (scale) giá trị tổn thất lên để ngăn gradient biến mất khi tính bằng `float16`."
    ],
    answer: 0,
    explanation: "Softmax và Crossentropy là những phép tính kém ổn định về số học (numerically unstable), nên thường chúng vẫn phải chạy bằng `float32` dù có bật Mixed Precision."
  },
  {
    type: "mcq",
    question: "Kỹ thuật đào tạo `float8` có đặc điểm gì nổi bật so với `float16`?",
    options: [
      "Nó làm mất đi quá nhiều thông tin nên chỉ khả thi đối với các mô hình cực lớn (LLM hàng tỷ tham số) chạy trên phần cứng chuyên dụng cao cấp như H100.",
      "Nó là giải pháp hoàn hảo cho mọi bài toán trên Kaggle, giúp mô hình tăng tốc gấp 4 lần ngay cả trên CPU của laptop thông thường.",
      "Nó cho phép thay đổi ngẫu nhiên cấu trúc mạng nơ-ron trong lúc huấn luyện để tìm ra số lớp tối ưu mà không cần dừng lại.",
      "Nó là định dạng do Apple phát minh, chỉ hoạt động trên thiết bị di động iOS để tăng tốc thuật toán nhận diện khuôn mặt."
    ],
    answer: 0,
    explanation: "float8 mất mát thông tin rất nhiều, nó yêu cầu sửa đổi sâu bên trong lớp (layer) và chỉ sinh lãi cho mô hình khổng lồ, thường được dùng trong đào tạo Foundation Models."
  }
];

export default questions;
