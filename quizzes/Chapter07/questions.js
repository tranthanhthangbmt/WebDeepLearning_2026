const quizData = [
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 1: Nguyên tắc thiết kế cốt lõi của API Keras là gì?",
    options: [
      "Luôn luôn yêu cầu người dùng viết toàn bộ mã nguồn từ đầu (from scratch).",
      "Tiết lộ dần dần về độ phức tạp (Progressive disclosure of complexity).",
      "Chỉ hỗ trợ một phương pháp xây dựng mô hình duy nhất để tránh nhầm lẫn.",
      "Ưu tiên hiệu suất tính toán tối đa bất chấp việc làm mã nguồn trở nên khó đọc."
    ],
    correctAnswer: 1,
    explanation: "Nguyên tắc thiết kế của Keras là 'Tiết lộ dần dần về độ phức tạp': Giúp người mới bắt đầu dễ dàng thực hiện các tác vụ đơn giản, nhưng vẫn cung cấp đường dẫn rõ ràng để các chuyên gia xử lý các tác vụ cực kỳ phức tạp (như viết vòng lặp tùy chỉnh)."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 2: Ba cách chính để xây dựng mô hình mạng nơ-ron trong Keras là gì?",
    options: [
      "Mô hình Tuần tự (Sequential), API Chức năng (Functional API), và Phân lớp Mô hình (Model Subclassing).",
      "Mô hình Cây quyết định (Decision Tree), API Tuyến tính, và Khối ResNet.",
      "Lớp Dense, Lớp Convolutional, và Lớp Recurrent (RNN).",
      "Hàm fit(), Hàm evaluate(), và Hàm predict()."
    ],
    correctAnswer: 0,
    explanation: "Đây là 3 cách chính trong phổ quy trình làm việc của Keras: từ dễ nhất (Sequential) đến linh hoạt nhất (Model Subclassing)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 3: Hạn chế lớn nhất của Mô hình Tuần tự (Sequential) trong Keras là gì?",
    options: [
      "Không thể huấn luyện bằng các bộ tối ưu hóa tiêu chuẩn như Adam hoặc SGD.",
      "Chỉ có thể có một đầu vào và một đầu ra duy nhất, xử lý theo chiều dọc xếp chồng.",
      "Yêu cầu phải viết mã nguồn phức tạp để kết nối các lớp với nhau.",
      "Không tương thích với các ứng dụng máy học chạy trên trình duyệt web."
    ],
    correctAnswer: 1,
    explanation: "Sequential bản chất giống như một danh sách (Python list). Nó không thể xử lý các kiến trúc đồ thị phức tạp như nhiều đầu vào (multi-input), nhiều đầu ra (multi-output), hoặc các luồng chia nhánh (residual connections)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 4: Điều gì sẽ xảy ra nếu bạn cố gắng gọi phương thức `model.summary()` trên một Mô hình Tuần tự (Sequential) vừa được khởi tạo và chứa vài lớp `Dense`, nhưng chưa cung cấp kích thước đầu vào?",
    options: [
      "Hàm sẽ in ra bản tóm tắt đầy đủ với số lượng tham số được ước tính mặc định.",
      "Sẽ xảy ra lỗi vì mô hình chưa được xây dựng (chưa tạo trọng số).",
      "Hệ thống sẽ tự động gán kích thước đầu vào mặc định là ma trận 28x28.",
      "Mô hình sẽ bỏ qua yêu cầu và tự động chuyển sang trạng thái huấn luyện."
    ],
    correctAnswer: 1,
    explanation: "Keras tạo các biến trọng số linh hoạt (Lazy execution). Các lớp chỉ được khởi tạo trọng số khi chúng biết chính xác kích thước đầu vào. Do đó, phải gọi `model.build(input_shape)` hoặc truyền một batch dữ liệu thực tế qua mô hình thì mới có thể `summary()`."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 5: Trong API Chức năng (Functional API), đối tượng trả về khi gọi `keras.Input(shape=(3,))` được gọi là gì?",
    options: [
      "Một ma trận (Tensor) chứa dữ liệu thực tế ngẫu nhiên để thử nghiệm.",
      "Một mảng NumPy chứa toàn bộ tập dữ liệu huấn luyện của mạng.",
      "Một Tensor Ký hiệu (Symbolic tensor) không chứa dữ liệu thực.",
      "Một đối tượng Bộ nhớ đệm (Cache Object) dùng để tối ưu hóa GPU."
    ],
    correctAnswer: 2,
    explanation: "Nó là một Symbolic Tensor. Nó không chứa dữ liệu thật (data) mà chỉ chứa các thông số kỹ thuật (hình dạng - shape, và kiểu dữ liệu - dtype) để Keras có thể xây dựng đồ thị toán học trước."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 6: Để khai báo hình dạng đầu vào trước khi thêm các lớp vào mô hình Sequential (nhằm dùng được summary() ngay lập tức), ta có thể sử dụng lớp keras.________.",
    blanks: ["Input", "input"],
    explanation: "Sử dụng model.add(keras.Input(shape=(...))) giúp Keras ngay lập tức biết hình dạng của luồng dữ liệu, cho phép nó xây dựng trọng số của các lớp tiếp theo ngay khi chúng được thêm vào."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 7: Vì sao API Chức năng (Functional API) lại rất phù hợp cho tác vụ Trích xuất đặc trưng (Feature extraction)?",
    options: [
      "Vì nó là phương pháp duy nhất có khả năng lưu trọng số ra file độc lập.",
      "Vì nó hình thành một cấu trúc đồ thị rõ ràng cho phép truy cập các lớp trung gian.",
      "Vì nó tự động nén dung lượng của các mạng nơ-ron học sâu để chạy nhanh hơn.",
      "Vì nó cung cấp các mô hình đã được huấn luyện sẵn (Pre-trained models) đi kèm."
    ],
    correctAnswer: 1,
    explanation: "Functional API xây dựng một đồ thị rõ ràng (Explicit Graph). Bạn có thể dễ dàng truy cập bất kỳ nút nào thông qua `model.layers`, lấy `layer.output`, và sử dụng output đó làm input cho một kiến trúc mô hình mới mà không cần huấn luyện lại từ đầu."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 8: Khác biệt cốt lõi giữa API Chức năng (Functional) và Phân lớp Mô hình (Model Subclassing) là gì?",
    options: [
      "API Chức năng sử dụng C++, trong khi Phân lớp Mô hình hoàn toàn sử dụng Python.",
      "Phân lớp Mô hình không thể được lưu dưới dạng file, còn API Chức năng thì có thể.",
      "API Chức năng là một cấu trúc dữ liệu đồ thị rõ ràng, Phân lớp là một hộp đen mã Python.",
      "Phân lớp Mô hình tự động tối ưu hóa siêu tham số, API Chức năng thì không."
    ],
    correctAnswer: 2,
    explanation: "Functional API tạo ra một biểu đồ (Graph) minh bạch, dễ dàng trực quan hóa qua `plot_model()`. Model Subclassing ẩn mọi logic toán học bên trong một phương thức Python `call()`, do đó nó hoàn toàn là một hộp đen linh hoạt nhưng Keras không thể vẽ sơ đồ cho nó."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 9: Khi nào bạn BẮT BUỘC phải sử dụng phương pháp Phân lớp Mô hình (Model Subclassing) thay vì API Chức năng?",
    options: [
      "Khi mô hình của bạn có từ hai đầu vào trở lên (ví dụ: ảnh và văn bản).",
      "Khi bạn cần chia sẻ một phần kiến trúc mạng với một dự án nhóm khác.",
      "Khi logic chuyển tiếp (forward pass) sử dụng vòng lặp đệ quy hoặc câu lệnh if-else động.",
      "Khi cấu hình máy tính của bạn không có card đồ họa (GPU) rời mạnh mẽ."
    ],
    correctAnswer: 2,
    explanation: "Bởi vì Functional API chỉ định nghĩa được một Đồ thị phi chu trình có hướng (Directed Acyclic Graph - DAG) tĩnh. Nếu luồng dữ liệu (forward pass) cần tính năng lập trình động (như đệ quy, vòng lặp for tùy ý, câu lệnh if thay đổi theo từng batch), bạn bắt buộc phải dùng Subclassing."
  },
  {
    type: "matching",
    difficulty: "Dễ",
    question: "Câu 10: Ghép nối phương pháp xây dựng mô hình Keras với đặc điểm chính xác nhất của nó:",
    pairs: [
      { left: "Mô hình Sequential", right: "Đơn giản nhất, giống như danh sách, chỉ có một luồng duy nhất." },
      { left: "API Chức năng (Functional)", right: "Giống khối xếp hình LEGO, đồ thị rõ ràng, phù hợp đa số tác vụ." },
      { left: "Phân lớp Mô hình (Subclassing)", right: "Linh hoạt nhất, phương thức call() là hộp đen, không thể vẽ đồ thị." }
    ],
    explanation: "Đó là 3 tùy chọn tương ứng với 3 mức độ kiểm soát trong nguyên lý Tiết lộ dần dần về độ phức tạp (Progressive disclosure of complexity) của Keras."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 11: Nếu bạn muốn sử dụng một Mô hình API Chức năng làm một khối con bên trong một Mô hình Phân lớp lớn hơn, điều này có khả thi không?",
    options: [
      "Hoàn toàn khả thi, tất cả các mô hình trong Keras đều có thể tương tác trơn tru với nhau.",
      "Không khả thi, vì hệ thống sẽ bị xung đột về cách biên dịch đồ thị (Graph compilation).",
      "Chỉ khả thi nếu Mô hình Chức năng đó đã được huấn luyện xong từ trước.",
      "Chỉ khả thi khi môi trường lập trình đang được thiết lập ở chế độ Eager Execution."
    ],
    correctAnswer: 0,
    explanation: "Trong Keras, mọi kiến trúc (Sequential, Functional, Subclassing) đều tương thích với nhau (Mix & Match). Bạn có thể dùng model A làm layer trong model B, bất chấp chúng được tạo ra bằng phương pháp nào."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 12: Vai trò của các lớp con (Subclass) của `keras.callbacks.Callback` trong quá trình gọi hàm `fit()` là gì?",
    options: [
      "Chúng trực tiếp định nghĩa cấu trúc của mạng nơ-ron thay cho API Chức năng.",
      "Chúng có khả năng quan sát trạng thái mô hình tại các mốc thời gian và chủ động thực hiện hành động.",
      "Chúng làm nhiệm vụ tự động tải dữ liệu từ internet khi bộ nhớ đầy.",
      "Chúng dịch các hàm tính toán đạo hàm từ Python sang ngôn ngữ C++ để tăng tốc."
    ],
    correctAnswer: 1,
    explanation: "Callbacks biến `model.fit()` từ một 'máy bay giấy' thành một 'drone thông minh'. Callback được gọi tại các mốc thời gian nhất định (như khi bắt đầu batch, kết thúc epoch) để quan sát (logs) và can thiệp (ví dụ: lưu mô hình, dừng sớm)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 13: Lệnh gọi lại (Callback) `EarlyStopping` thường được sử dụng nhằm giải quyết vấn đề gì trong học máy?",
    options: [
      "Tránh việc máy tính bị quá tải nhiệt khi huấn luyện các mạng nơ-ron cực lớn.",
      "Ngăn chặn mô hình rơi vào trạng thái Overfitting bằng cách dừng huấn luyện đúng lúc.",
      "Loại bỏ các điểm dữ liệu bị nhiễu và sai lệch ra khỏi tập dữ liệu kiểm tra.",
      "Tự động điều chỉnh kích thước batch (batch size) sao cho tốc độ hội tụ nhanh hơn."
    ],
    correctAnswer: 1,
    explanation: "Dừng sớm (Early Stopping) giúp ngắt quá trình huấn luyện khi nhận thấy hàm mất mát trên tập kiểm chứng (validation loss) bắt đầu có dấu hiệu tăng lên (dấu hiệu của Overfitting), giúp tiết kiệm thời gian huấn luyện thừa thãi."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 14: Đối số `save_best_only=True` trong Callback `ModelCheckpoint` có ý nghĩa gì?",
    options: [
      "Nó tự động sao lưu toàn bộ các mô hình từng được tạo ra vào một thư mục đám mây.",
      "Nó chỉ lưu lại một phiên bản của mô hình nếu hiệu suất của phiên bản đó tốt hơn kỷ lục trước đó.",
      "Nó ngăn chặn việc mô hình chạy thử nghiệm nếu kết quả dự đoán chưa vượt qua mốc 90%.",
      "Nó đảm bảo mô hình chỉ lưu lại phần trọng số (weights) thay vì toàn bộ cả cấu trúc (architecture)."
    ],
    correctAnswer: 1,
    explanation: "Khi được thiết lập, Keras sẽ không lưu mô hình tại cuối MỖI epoch một cách mù quáng, mà chỉ lưu (ghi đè) file nếu chỉ số đang theo dõi (ví dụ val_loss) đạt kết quả tốt nhất từ đầu quá trình đào tạo."
  },
  {
    type: "fill_blank",
    difficulty: "Dễ",
    question: "Câu 15: Để tạo một số liệu tùy chỉnh (Custom Metric) hoạt động tương tự như Accuracy, bạn cần tạo một lớp kế thừa từ lớp `keras.________.Metric`.",
    blanks: ["metrics"],
    explanation: "Các thước đo đánh giá tùy chỉnh được xây dựng bằng cách subclassing `keras.metrics.Metric`, trong đó bạn phải định nghĩa `update_state()`, `result()`, và `reset_state()`."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 16: Trong quá trình xây dựng một Metric tùy chỉnh, biến trạng thái bên trong hàm `update_state()` CÓ được cập nhật thông qua thuật toán Lan truyền ngược (Backpropagation) không?",
    options: [
      "Có, chúng được cập nhật tự động như các trọng số (weights) của nơ-ron.",
      "Không, chúng là các biến trạng thái thủ công do lập trình viên tự tính toán.",
      "Có, nhưng chỉ khi bạn bật chế độ eager execution của TensorFlow.",
      "Không, vì chúng bị vô hiệu hóa khi mô hình được triển khai qua REST API."
    ],
    correctAnswer: 1,
    explanation: "Biến trạng thái của Metric không phải là tham số có thể huấn luyện (non-trainable weights). Lập trình viên phải TỰ VIẾT logic tính toán cộng/dồn trực tiếp trong phương thức `update_state()`."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 17: Trong quá trình thiết kế một Callback tùy chỉnh, phương thức nào sau đây sẽ được gọi ngay sau khi mô hình tính toán xong một Mini-batch?",
    options: [
      "on_epoch_end",
      "on_batch_begin",
      "on_train_end",
      "on_batch_end"
    ],
    correctAnswer: 3,
    explanation: "Khi xử lý xong một batch, callback sẽ gọi `on_batch_end(batch_index, logs)`. Đối tượng `logs` tại mốc này sẽ chứa hàm mất mát và các chỉ số đo lường tính cho lô dữ liệu vừa qua."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 18: Công cụ `TensorBoard` là gì và nó mang lại lợi ích chính nào?",
    options: [
      "Một trình tối ưu hóa tiên tiến được tạo ra để thay thế hoàn toàn thuật toán Adam.",
      "Một công cụ trực quan hóa cung cấp biểu đồ trực tiếp về hàm mất mát, mạng nơ-ron và trọng số.",
      "Một định dạng lưu trữ file độc quyền của Keras để nén các mạng nơ-ron sâu.",
      "Một hệ quản trị cơ sở dữ liệu phân tán để chạy quá trình học trên nhiều GPU."
    ],
    correctAnswer: 1,
    explanation: "TensorBoard là giao diện người dùng dựa trên trình duyệt. Nó hỗ trợ trực quan hóa đường cong độ chính xác, phân tích profile hiệu suất, biểu diễn hình dáng mô hình toán học và kiểm tra độ dốc."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 19: Tại sao đối số `training=True` hoặc `training=False` lại vô cùng quan trọng khi gọi mô hình trong một vòng lặp huấn luyện tùy chỉnh (custom training loop)?",
    options: [
      "Vì nếu bỏ sót đối số này, ngôn ngữ Python sẽ tự động báo lỗi cú pháp (Syntax error).",
      "Vì nó thông báo cho cơ sở dữ liệu biết khi nào nên tải thêm luồng dữ liệu mới.",
      "Vì một số lớp như Dropout và BatchNormalization có hành vi hoàn toàn khác nhau trong và sau khi huấn luyện.",
      "Vì mô hình sẽ chạy cực kỳ chậm và tốn dung lượng RAM nếu không bật cờ này."
    ],
    correctAnswer: 2,
    explanation: "Lớp Dropout chỉ ngẫu nhiên tắt nơ-ron khi đang HUẤN LUYỆN. Khi SUY LUẬN (inference), lớp này phải tắt để bảo toàn độ chính xác. Việc gọi `model(inputs, training=True)` giúp các lớp có trạng thái như Dropout hoạt động đúng logic."
  },
  {
    type: "sorting",
    difficulty: "Khó",
    question: "Câu 20: Sắp xếp các bước cơ bản để triển khai một vòng lặp đào tạo tùy chỉnh (Custom Training Loop) cấp thấp bằng mã giả:",
    steps: [
      "Gọi forward pass: tính toán đầu ra của mô hình (predictions).",
      "Tính giá trị Hàm mất mát (Loss) dựa trên Predictions và Targets.",
      "Sử dụng GradientTape để tính toán độ dốc (Gradients).",
      "Sử dụng Optimizer để áp dụng độ dốc, cập nhật trọng số (Weights)."
    ],
    explanation: "Đây là 4 bước chuẩn của quá trình Lan truyền ngược (Backpropagation): Chuyển tiếp (forward) -> Tính Lỗi (Loss) -> Tìm Đạo hàm (Grads) -> Cập nhật (Apply grads)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 21: Khi viết vòng huấn luyện tùy chỉnh, việc truy xuất `model.trainable_weights` khác gì với `model.weights`?",
    options: [
      "Chúng hoàn toàn giống nhau, đây chỉ là 2 cách đặt tên hàm khác nhau trong Keras.",
      "trainable_weights chỉ bao gồm những trọng số bị ảnh hưởng bởi quá trình đạo hàm (lan truyền ngược).",
      "weights chỉ chứa thông tin cấu hình phần cứng mạng nơ-ron, còn trainable_weights là dữ liệu ảnh.",
      "trainable_weights là những biến trạng thái tĩnh không được phép thay đổi suốt quá trình huấn luyện."
    ],
    correctAnswer: 1,
    explanation: "Các mạng nơ-ron chứa các tham số được học bằng gradient (Trainable) và các trạng thái lưu trữ dữ liệu tính toán tích lũy nhưng không dùng gradient (Non-trainable, như tham số moving mean của BatchNormalization). `model.weights` trả về cả 2."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 22: Phương thức nào sau đây của Keras cho phép ghi đè hoàn toàn toàn bộ logic bên trong hàm `fit()` (cho 1 batch học)?",
    options: [
      "Phương thức call()",
      "Phương thức fit_batch()",
      "Phương thức train_step()",
      "Phương thức custom_backprop()"
    ],
    correctAnswer: 2,
    explanation: "Trong Keras 3, bạn có thể tạo một lớp kế thừa từ `keras.Model` và ghi đè phương thức `train_step()`. Hàm `fit()` tích hợp sẵn sẽ tự động gọi phương thức `train_step()` tùy chỉnh của bạn cho mỗi lô (batch)."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 23: Nếu bạn đang viết một `train_step()` tùy chỉnh và quên mất việc gọi `metric.update_state()` trong vòng lặp, hậu quả là gì?",
    options: [
      "Chương trình sẽ ngay lập tức treo và đóng (Crash) do thiếu dữ liệu Log.",
      "Hàm mất mát sẽ hội tụ về vô cực, dẫn đến vấn đề nổ gradient.",
      "Trọng số của mạng sẽ ngừng cập nhật và thuật toán lan truyền ngược thất bại.",
      "Các giá trị độ chính xác và số liệu đo lường hiển thị trên màn hình sẽ đứng im."
    ],
    correctAnswer: 3,
    explanation: "Metric độc lập hoàn toàn với quá trình Gradient (Loss function). Nếu bạn không update state của metric, mô hình vẫn học và trọng số vẫn cập nhật bình thường, nhưng thông báo tiến độ (progress bar) hiển thị Accuracy sẽ không thay đổi."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 24: Keras thường được ví von là '______ của deep learning' do triết lý thiết kế đa mô hình (multiparadigm) và tính linh hoạt của nó.",
    options: [
      "Ngôn ngữ C++",
      "Hệ điều hành Windows",
      "Ngôn ngữ Python",
      "Định dạng JSON"
    ],
    correctAnswer: 2,
    explanation: "Sách ví Keras như là Python của Deep Learning, bởi nó hỗ trợ nhiều trường phái viết code (như cách Python hỗ trợ Object-oriented và Functional), phù hợp từ người mới học đến chuyên gia tinh chỉnh."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 25: Hàm `plot_model()` hỗ trợ việc in ra màn hình cấu trúc đồ thị của mô hình. Tại sao hàm này có thể thất bại khi xử lý Mô hình Phân lớp (Subclassing Model)?",
    options: [
      "Do mô hình Phân lớp không hỗ trợ kiến trúc có nhiều lớp Dense liên tiếp.",
      "Do phương thức call() ẩn toàn bộ logic kết nối trong đoạn mã Python thô.",
      "Do máy tính thiếu thư viện vẽ đồ thị mã nguồn mở bắt buộc của Python.",
      "Do quá trình Phân lớp đòi hỏi mô hình phải được lưu dưới định dạng PDF."
    ],
    correctAnswer: 1,
    explanation: "Như đã nói, Functional API giữ một đồ thị rõ ràng (Explicit Data Structure). Model Subclassing ẩn nó trong một hàm (Black Box), do đó Keras không có thông tin về đồ thị để vẽ."
  },
  {
    type: "matching",
    difficulty: "Khó",
    question: "Câu 26: Ghép nối Backend với các hàm đặc trưng để lấy độ dốc (Gradients) trong một vòng lặp huấn luyện cấp thấp:",
    pairs: [
      { left: "TensorFlow", right: "tape.gradient(loss, weights) trong khối with tf.GradientTape()" },
      { left: "PyTorch", right: "loss.backward()" },
      { left: "JAX", right: "jax.value_and_grad()" }
    ],
    explanation: "Đây là 3 cú pháp nền tảng đại diện cho 3 framework lớn nhất hiện nay. Keras 3 cho phép bạn dùng linh hoạt cú pháp của bất kỳ framework nào."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 27: Khi sử dụng `ModelCheckpoint`, nếu bạn chỉnh đối số `monitor='val_accuracy'`, mô hình sẽ ưu tiên lưu lại khi điều kiện gì xảy ra?",
    options: [
      "Khi độ chính xác trên tập dữ liệu huấn luyện (Training) tăng cao hơn.",
      "Khi hàm mất mát trên tập dữ liệu kiểm chứng (Validation loss) giảm đột ngột.",
      "Khi độ chính xác trên tập dữ liệu kiểm chứng (Validation accuracy) đạt đỉnh cao mới.",
      "Khi thiết bị máy tính có dấu hiệu bị cạn kiệt dung lượng bộ nhớ tạm (RAM)."
    ],
    correctAnswer: 2,
    explanation: "Tham số `monitor='val_accuracy'` báo cho hệ thống kiểm tra sự cải thiện (điểm cao hơn) của độ chính xác trên tập kiểm chứng để quyết định việc lưu lại Checkpoint."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 28: Thuộc tính `model.layers` bên trong API Chức năng trả về cái gì?",
    options: [
      "Một mảng NumPy chứa toàn bộ giá trị trọng số của mô hình.",
      "Một danh sách các Lớp (Layers) tạo thành mạng nơ-ron đó.",
      "Chuỗi ký tự in ra cấu trúc thông tin về thiết kế của mô hình.",
      "Một con số nguyên duy nhất biểu thị số lượng tầng mạng."
    ],
    correctAnswer: 1,
    explanation: "`model.layers` là một List của Python chứa đối tượng Layer tương ứng. Bạn có thể duyệt qua danh sách này, trích xuất `layer.output` để tái sử dụng."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 29: Nếu bạn truyền danh sách Dictionary thay vì List thông thường trong quá trình huấn luyện `model.fit()` (API Chức năng đa đầu vào), điều kiện nào sau đây là BẮT BUỘC?",
    options: [
      "Tất cả các phần tử phải có kiểu dữ liệu là một số thập phân (float32).",
      "Tất cả các lớp đầu vào trong mô hình đều phải là lớp Dense.",
      "Tên các khóa (Keys) của từ điển phải khớp chính xác với tên của các lớp Input.",
      "Kích thước khối (Batch size) phải được cố định hoàn toàn ở ngưỡng 128."
    ],
    correctAnswer: 2,
    explanation: "Khi truyền dict `{ \"tieu_de\": text_data }`, Keras sẽ tìm đối tượng `keras.Input(name=\"tieu_de\")` để bơm dữ liệu vào. Nếu tên không khớp, Keras không biết luồng dữ liệu thuộc về cổng nào."
  },
  {
    type: "fill_blank",
    difficulty: "Khó",
    question: "Câu 30: Trong API Keras, phương thức được thiết kế để tạo nhanh trọng số của mô hình (thay vì phải gọi suy luận bằng dữ liệu giả) dựa trên kích thước đầu vào được gọi là model.________(input_shape).",
    blanks: ["build"],
    explanation: "Phương thức `model.build((None, shape))` chủ động yêu cầu Keras phân bổ tài nguyên bộ nhớ và khởi tạo trọng số mà không cần truyền tensor dữ liệu."
  }
];

export default quizData;
