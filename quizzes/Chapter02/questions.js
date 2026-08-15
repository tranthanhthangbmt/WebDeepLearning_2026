const quizData = [
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 1: MNIST dataset thường được ví như bài toán nào trong lĩnh vực học sâu?",
    options: [
      "Bài toán phức tạp nhất để đánh giá hiệu suất phần cứng.",
      "Chương trình \"Hello World\" dùng để xác minh các thuật toán hoạt động như mong đợi.",
      "Bài toán phân loại ảnh màu 3D có độ phân giải cao.",
      "Thử nghiệm đầu tiên cho các thuật toán học tăng cường."
    ],
    correctAnswer: 1,
    explanation: "MNIST là tập dữ liệu phân loại chữ số viết tay đơn giản. Giống như việc in ra \"Hello World\" khi học ngôn ngữ lập trình mới, giải quyết MNIST là bước đầu tiên để đảm bảo rằng mạng lưới thần kinh đang được thiết lập và hoạt động đúng cách."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 2: Trong Keras, vai trò chính của lớp (layer) là gì?",
    options: [
      "Trích xuất các biểu diễn (representations) có ý nghĩa hơn từ dữ liệu đầu vào.",
      "Lưu trữ toàn bộ tập dữ liệu huấn luyện vào bộ nhớ cache.",
      "Hiển thị giao diện đồ họa (GUI) cho quá trình huấn luyện.",
      "Tính toán đạo hàm trực tiếp của hàm mất mát mà không cần dữ liệu."
    ],
    correctAnswer: 0,
    explanation: "Layer giống như một bộ lọc dữ liệu. Nó tiếp nhận dữ liệu đầu vào và chuyển đổi thành một biểu diễn mới, giúp cho bài toán đang cần giải quyết trở nên dễ dàng hơn. Deep learning chính là việc xếp chồng các layer này lại để tinh cất dữ liệu dần dần."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 3: Tensor rank-0 (0D tensor) còn được gọi là gì?",
    options: [
      "Vector",
      "Ma trận (Matrix)",
      "Vô hướng (Scalar)",
      "Mảng đa chiều (Multidimensional array)"
    ],
    correctAnswer: 2,
    explanation: "Một tensor chỉ chứa đúng một con số duy nhất được gọi là scalar (vô hướng). Trong NumPy, một số float32 hoặc float64 đứng độc lập chính là một scalar tensor, có số chiều (ndim) bằng 0."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 4: Kích thước (shape) của một tensor ảnh màu (RGB) thông thường sử dụng chuẩn 'channels-last' trong Keras là gì?",
    options: [
      "(samples, channels, height, width)",
      "(samples, height, width, channels)",
      "(height, width, channels, samples)",
      "(channels, samples, height, width)"
    ],
    correctAnswer: 1,
    explanation: "Chuẩn channels-last (mặc định của Keras/TensorFlow) đặt trục kênh màu ở cuối cùng: (samples, height, width, color_depth). PyTorch lại dùng channels-first: (samples, color_depth, height, width)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 5: Phát biểu nào sau đây mô tả ĐÚNG NHẤT về trục mẫu (samples axis / batch axis)?",
    options: [
      "Là trục cuối cùng của mọi tensor dữ liệu.",
      "Là trục đại diện cho các đặc trưng (features) của từng mẫu.",
      "Là trục đầu tiên (trục 0) trong tất cả các data tensor dùng trong học sâu.",
      "Chỉ xuất hiện trong dữ liệu chuỗi thời gian."
    ],
    correctAnswer: 2,
    explanation: "Theo quy ước trong học sâu, trục đầu tiên (axis 0) luôn được dùng để chứa các mẫu dữ liệu độc lập (ví dụ: các bức ảnh khác nhau). Khi chia dữ liệu thành các lô (batch), trục này được gọi là batch axis."
  },
  {
    type: "fill_blank",
    difficulty: "Dễ",
    question: "Câu 6: Một tensor chứa mảng các con số được gọi là vector, hay tensor hạng (rank) mấy?",
    blanks: ["1", "một", "one"],
    explanation: "Vector là mảng một chiều, do đó nó có đúng 1 trục. Nó được gọi là tensor hạng 1 (rank-1 tensor)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 7: Một video clip thường được lưu trữ dưới dạng tensor hạng mấy?",
    options: [
      "Hạng 3 (Rank-3)",
      "Hạng 4 (Rank-4)",
      "Hạng 5 (Rank-5)",
      "Hạng 6 (Rank-6)"
    ],
    correctAnswer: 2,
    explanation: "Video được lưu trữ dưới dạng Rank-5 tensor với cấu trúc: (samples, frames, height, width, channels). Mỗi sample là một video, mỗi video gồm nhiều frame, mỗi frame là một ảnh màu 3D."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 8: Khái niệm 'Broadcasting' (phát sóng) trong phép toán tensor là gì?",
    options: [
      "Kỹ thuật truyền dữ liệu qua mạng LAN để huấn luyện phân tán trên nhiều GPU.",
      "Quá trình thêm các trục mới (broadcast axes) vào tensor nhỏ hơn và lặp lại dữ liệu để khớp với shape của tensor lớn hơn trước khi tính toán.",
      "Việc chuyển đổi ngẫu nhiên các giá trị của ma trận để tăng cường dữ liệu (data augmentation).",
      "Tính năng gửi kết quả của hàm mất mát cho trình tối ưu hóa (optimizer)."
    ],
    correctAnswer: 1,
    explanation: "Broadcasting cho phép thực hiện các phép toán element-wise giữa hai tensor có shape khác nhau. Tensor nhỏ hơn sẽ được tự động 'mở rộng' (thêm trục ảo và lặp lại dữ liệu) để khớp với tensor lớn hơn."
  },
  {
    type: "sorting",
    difficulty: "Trung bình",
    question: "Câu 9: Sắp xếp các bước chuẩn bị dữ liệu MNIST từ dữ liệu thô đến khi đưa vào mô hình:",
    steps: [
      "Tải dữ liệu bằng hàm mnist.load_data()",
      "Reshape mảng 3D (60000, 28, 28) thành mảng 2D (60000, 28 * 28)",
      "Chuyển đổi kiểu dữ liệu thành float32",
      "Scale (chuẩn hóa) các giá trị pixel vào khoảng [0, 1] bằng cách chia cho 255"
    ],
    explanation: "Quy trình chuẩn: Tải dữ liệu -> Reshape thành vector phẳng (vì dùng Dense layer) -> Đổi kiểu sang float32 -> Chuẩn hóa về [0, 1]."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 10: Phép toán `relu(x)` thực chất đang thực hiện phép tính toán học nào?",
    options: [
      "Bình phương của x",
      "Lấy giá trị lớn nhất giữa x và 0: max(x, 0)",
      "Trả về giá trị tuyệt đối của x: abs(x)",
      "Tính đạo hàm của x"
    ],
    correctAnswer: 1,
    explanation: "Hàm kích hoạt ReLU (Rectified Linear Unit) sẽ giữ nguyên các giá trị dương và biến tất cả các giá trị âm thành 0. Công thức đơn giản của nó là max(x, 0)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 11: Gradient của một hàm đa biến (tensor operation) có ý nghĩa hình học là gì?",
    options: [
      "Độ dài đường chéo của ma trận trọng số.",
      "Điểm mà tại đó hàm số luôn luôn bằng 0.",
      "Hướng dốc nhất (steepest ascent) của bề mặt hàm số tại một điểm cho trước.",
      "Giá trị trung bình của tất cả các phần tử trong tensor."
    ],
    correctAnswer: 2,
    explanation: "Gradient đại diện cho độ dốc của hàm đa biến. Nó chỉ ra hướng mà hàm số tăng nhanh nhất. Để giảm thiểu hàm mất mát, chúng ta di chuyển các trọng số theo hướng NGƯỢC LẠI với gradient."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 12: Đâu là đặc điểm KHÁC BIỆT cơ bản nhất giữa phép nhân phần tử (element-wise product) và phép nhân tensor (tensor dot product)?",
    options: [
      "Phép nhân phần tử chỉ dùng được cho ma trận, tensor dot product dùng cho vector.",
      "Phép nhân phần tử nhân các phần tử cùng vị trí (*), trong khi tensor dot product (dấu chấm) kết hợp các phần tử để tạo ra hình dạng (shape) hoàn toàn mới.",
      "Phép nhân phần tử tiêu tốn nhiều bộ nhớ GPU hơn.",
      "Tensor dot product có thể tự động broadcasting, còn phép nhân phần tử thì không."
    ],
    correctAnswer: 1,
    explanation: "Toán tử `*` trong numpy thực hiện nhân từng phần tử tương ứng. Toán tử `np.dot` tính toán tổng các tích theo hàng và cột, kết hợp các tensor lại và thay đổi hoàn toàn hình dạng (shape) của kết quả đầu ra."
  },
  {
    type: "matching",
    difficulty: "Trung bình",
    question: "Câu 13: Ghép nối các khái niệm trong biên dịch mô hình (compilation) với ý nghĩa tương ứng:",
    pairs: [
      { left: "Loss function (Hàm mất mát)", right: "Thước đo sự chênh lệch giữa dự đoán và nhãn thực tế." },
      { left: "Optimizer (Trình tối ưu hóa)", right: "Thuật toán dùng để cập nhật trọng số mạng dựa trên gradient." },
      { left: "Metrics (Đo lường)", right: "Tiêu chí theo dõi trong lúc huấn luyện (VD: độ chính xác - accuracy)." },
      { left: "Epoch", right: "Một vòng lặp huấn luyện đi qua toàn bộ dữ liệu." }
    ],
    explanation: "Hàm mất mát đánh giá mức độ sai lệch, Optimizer cập nhật trọng số để giảm thiểu sai lệch đó, còn Metrics chỉ dùng để con người dễ theo dõi (không dùng để tính gradient)."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 14: Tại sao chúng ta sử dụng mini-batch gradient descent thay vì tính toán trên toàn bộ dữ liệu một lúc?",
    options: [
      "Vì hàm mất mát không thể tính được trên toàn bộ dữ liệu.",
      "Tính toán trên toàn bộ dữ liệu quá tốn kém bộ nhớ và thời gian cho mỗi bước cập nhật, đồng thời mini-batch giúp thêm tính ngẫu nhiên tránh tối ưu cục bộ.",
      "Vì ngôn ngữ Python không hỗ trợ mảng có kích thước quá 60,000 phần tử.",
      "Để ngăn chặn việc tính toán đạo hàm bị lỗi chia cho 0."
    ],
    correctAnswer: 1,
    explanation: "Tính toán gradient trên toàn bộ dữ liệu (Batch Gradient Descent) đòi hỏi bộ nhớ rất lớn và cập nhật rất chậm. Mini-batch chia nhỏ dữ liệu giúp cập nhật trọng số nhanh hơn nhiều lần và độ nhiễu ngẫu nhiên giúp thoát khỏi local minima."
  },
  {
    type: "fill_blank",
    difficulty: "Khó",
    question: "Câu 15: Thuật toán chính được dùng để tính toán gradient của tất cả các lớp trong một mạng lưới thần kinh phức tạp được gọi là ________.",
    blanks: ["Backpropagation", "Lan truyền ngược", "Backprop"],
    explanation: "Backpropagation (Lan truyền ngược) áp dụng quy tắc chuỗi (chain rule) của vi phân để tính toán gradient từ lớp cuối cùng (hàm mất mát) ngược dần về các lớp đầu tiên."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 16: Dữ liệu chuỗi thời gian (Timeseries) thường được lưu trữ dưới dạng Tensor hạng mấy?",
    options: [
      "Hạng 1",
      "Hạng 2",
      "Hạng 3",
      "Hạng 4"
    ],
    correctAnswer: 2,
    explanation: "Dữ liệu chuỗi thời gian luôn cần thêm một trục thời gian (timesteps). Cấu trúc của nó là (samples, timesteps, features), tức là một Rank-3 tensor."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 17: Tại sao hiện tượng Overfitting (quá khớp) lại xảy ra?",
    options: [
      "Mô hình quá đơn giản, không đủ sức chứa để học các mẫu dữ liệu.",
      "Mô hình học 'thuộc lòng' các nhiễu và chi tiết cụ thể của tập huấn luyện, khiến nó hoạt động kém đi trên dữ liệu mới chưa từng thấy.",
      "Tốc độ học (learning rate) được thiết lập quá lớn khiến gradient phân kỳ.",
      "Số lượng epoch quá ít, mô hình chưa kịp hội tụ."
    ],
    correctAnswer: 1,
    explanation: "Overfitting xảy ra khi mô hình ghi nhớ quá mức tập dữ liệu huấn luyện (bao gồm cả nhiễu), làm mất đi khả năng tổng quát hóa (generalization) trên dữ liệu kiểm thử thực tế."
  },
  {
    type: "sorting",
    difficulty: "Khó",
    question: "Câu 18: Sắp xếp các bước chính xác của vòng lặp huấn luyện (Training Loop) trong Gradient Descent:",
    steps: [
      "Lấy ra một lô (batch) dữ liệu huấn luyện và nhãn tương ứng.",
      "Chạy mô hình (forward pass) để đưa ra dự đoán cho lô dữ liệu.",
      "Tính toán hàm mất mát (loss) so sánh dự đoán với nhãn thực tế.",
      "Tính toán gradient của hàm mất mát đối với tất cả trọng số của mô hình (backward pass).",
      "Cập nhật trọng số ngược hướng với gradient (weight -= step * gradient)."
    ],
    explanation: "Luồng hoạt động chuẩn: Lấy dữ liệu -> Lan truyền tiến (Forward) -> Tính lỗi -> Lan truyền ngược (Backward) -> Cập nhật trọng số."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 19: Khái niệm 'Momentum' trong các trình tối ưu hóa (optimizer) có tác dụng giải quyết vấn đề gì?",
    options: [
      "Giải quyết vấn đề tốc độ tải dữ liệu từ ổ cứng lên GPU.",
      "Tránh việc thuật toán bị kẹt ở các điểm cực tiểu cục bộ (local minima) bằng cách tính đến các cập nhật gradient trước đó.",
      "Giảm thiểu kích thước của mô hình để chạy trên điện thoại di động.",
      "Chuyển đổi ảnh màu sang ảnh xám để tăng tốc xử lý."
    ],
    correctAnswer: 1,
    explanation: "Momentum giống như một quả bóng lăn xuống đồi. Nếu đồi có một hố nhỏ (cực tiểu cục bộ), đà (momentum) từ việc lăn trước đó sẽ giúp quả bóng vượt qua hố nhỏ đó để tiếp tục lăn xuống đáy sâu nhất."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 20: Kích thước đầu vào mong đợi của lớp `layers.Dense(512)` đối với dữ liệu MNIST là mảng 2D (batch_size, 784). Tính chất toán học đằng sau lớp này là gì?",
    options: [
      "Chỉ là phép cộng hai tensor với nhau.",
      "Chỉ là phép áp dụng hàm kích hoạt phi tuyến tính.",
      "output = relu(dot(input, W) + b)",
      "Tính toán đạo hàm của input."
    ],
    correctAnswer: 2,
    explanation: "Lớp Dense thực hiện phép nhân tensor (dot product) giữa đầu vào và ma trận trọng số W, cộng thêm vector độ lệch (bias) b, và cuối cùng áp dụng hàm kích hoạt (như relu)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 21: Thuộc tính `.ndim` trong mảng NumPy dùng để làm gì?",
    options: [
      "Đếm tổng số lượng phần tử có trong mảng.",
      "Xác định xem mảng có chứa giá trị NaN không.",
      "Trả về số lượng trục (số chiều/rank) của tensor.",
      "Trả về kiểu dữ liệu (dtype) của các phần tử."
    ],
    correctAnswer: 2,
    explanation: "Trong NumPy, .ndim (number of dimensions) trả về hạng (rank) của tensor, tức là số lượng trục. Ví dụ mảng 2D có ndim = 2, mảng 3D có ndim = 3."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 22: Phép toán Tensor Slice `my_slice = train_images[:, 14:, 14:]` đối với MNIST dataset (ảnh 28x28) thực hiện điều gì?",
    options: [
      "Chọn ra 14 bức ảnh đầu tiên trong tập dữ liệu.",
      "Cắt lấy góc dưới cùng bên phải (kích thước 14x14 pixel) của tất cả các bức ảnh.",
      "Cắt lấy góc trên cùng bên trái của tất cả bức ảnh.",
      "Bỏ đi 14 pixel đầu tiên của mọi bức ảnh."
    ],
    correctAnswer: 1,
    explanation: "Dấu `:` ở trục đầu tiên lấy toàn bộ mẫu. `14:` lấy từ pixel số 14 đến hết. Vì gốc tọa độ ảnh nằm ở góc trên-trái, lấy từ pixel số 14 (nửa sau) cho cả chiều cao và chiều rộng đồng nghĩa với việc cắt góc dưới-phải của ảnh."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 23: Hàm kích hoạt dùng ở lớp cuối cùng cho bài toán phân loại đa lớp (multi-class) như phân loại 10 chữ số thường là ________.",
    blanks: ["softmax", "Soft max"],
    explanation: "Softmax biến đổi đầu ra thành một phân phối xác suất: tất cả các giá trị cộng lại bằng 1, và mỗi giá trị nằm trong khoảng [0, 1]. Rất lý tưởng cho phân loại đa lớp."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 24: Ký hiệu đạo hàm dây chuyền (Chain rule) áp dụng trong đồ thị tính toán có thể phát biểu đơn giản như thế nào?",
    options: [
      "Đạo hàm của tổng bằng tổng các đạo hàm: f(x+y)' = f'(x) + f'(y)",
      "Đạo hàm của hàm hợp là tích của các đạo hàm cục bộ: f(g(x))' = f'(g(x)) * g'(x)",
      "Đạo hàm của hàm hằng số luôn bằng 0.",
      "Đạo hàm luôn làm giảm giá trị của hàm."
    ],
    correctAnswer: 1,
    explanation: "Chain rule (Quy tắc chuỗi) phát biểu rằng đạo hàm của hàm hợp bằng tích các đạo hàm của các hàm thành phần. Đây chính là xương sống của thuật toán Backpropagation."
  },
  {
    type: "matching",
    difficulty: "Dễ",
    question: "Câu 25: Ghép nối tên gọi Tensor với mô tả hình dáng số chiều:",
    pairs: [
      { left: "Rank-0 Tensor", right: "Một con số duy nhất (Scalar)" },
      { left: "Rank-1 Tensor", right: "Mảng các con số (Vector)" },
      { left: "Rank-2 Tensor", right: "Mảng 2 chiều gồm các hàng và cột (Ma trận)" },
      { left: "Rank-3 Tensor", right: "Hình lập phương các con số (Ví dụ: Chuỗi thời gian)" }
    ],
    explanation: "Đây là định nghĩa cơ bản về số chiều (rank) của Tensor. Scalar = 0D, Vector = 1D, Matrix = 2D."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 26: Việc sử dụng GPU thay vì CPU mang lại lợi ích cốt lõi nào cho mạng lưới thần kinh?",
    options: [
      "Ngăn chặn việc mô hình bị Overfitting.",
      "Xử lý các phép tính logic phức tạp như lệnh If-Else nhanh hơn.",
      "Khả năng thực thi song song hàng nghìn phép toán tensor nhỏ lẻ cùng một lúc, tăng tốc độ tính toán gấp nhiều lần.",
      "Tự động tìm kiếm tập dữ liệu trên internet."
    ],
    correctAnswer: 2,
    explanation: "Mạng lưới thần kinh thực chất là hàng tỷ phép tính nhân ma trận cộng dồn. GPU có cấu trúc hàng nghìn lõi (core) thiết kế chuyên biệt để tính toán các phép toán song song này cực kỳ nhanh."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 27: Giá trị `loss` và `accuracy` hiển thị trong quá trình gọi hàm `fit()` được tính toán trên tập dữ liệu nào?",
    options: [
      "Trên tập dữ liệu kiểm thử (Test set).",
      "Trên tập dữ liệu huấn luyện (Training set).",
      "Chỉ trên 1 mẫu dữ liệu ngẫu nhiên.",
      "Trên cả hai tập Training và Test trộn lẫn."
    ],
    correctAnswer: 1,
    explanation: "Trong vòng lặp fit(), mặc định các chỉ số hiển thị là kết quả tính toán trên tập dữ liệu huấn luyện hiện tại. Để đánh giá trên tập kiểm thử, ta phải dùng `evaluate()` hoặc truyền `validation_data`."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 28: Đạo hàm (Derivative) của một hàm số tại một điểm x có ý nghĩa gì đối với việc học của mô hình?",
    options: [
      "Cho biết mô hình đã đạt đến độ chính xác tối đa hay chưa.",
      "Xác định độ lớn của learning_rate.",
      "Cung cấp hệ số góc của tiếp tuyến tại x, từ đó biết nên di chuyển x sang trái hay phải để làm giảm giá trị của hàm.",
      "Thay đổi cấu trúc của mạng (thêm hoặc bớt layer)."
    ],
    correctAnswer: 2,
    explanation: "Đạo hàm cho ta biết sự thay đổi cục bộ của hàm số. Nếu đạo hàm dương, di chuyển x sang trái (trừ đi) sẽ làm hàm số giảm. Đây là cốt lõi của tối ưu hóa."
  },
  {
    type: "fill_blank",
    difficulty: "Khó",
    question: "Câu 29: Tham số kiểm soát mức độ mà trọng số sẽ được cập nhật trong mỗi bước gradient descent được gọi là tốc độ học, hay tiếng Anh là ________.",
    blanks: ["learning rate", "learning_rate"],
    explanation: "Learning Rate (tốc độ học) là siêu tham số quan trọng nhất. Nếu quá nhỏ, mô hình học quá chậm; nếu quá lớn, mô hình có thể không bao giờ hội tụ."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 30: Một mảng NumPy có shape là `(10000, 28, 28)`. Nếu dùng hàm `.reshape((10000, 28 * 28))` thì kết quả trả về là tensor hạng mấy?",
    options: [
      "Hạng 1",
      "Hạng 2",
      "Hạng 3",
      "Hạng 4"
    ],
    correctAnswer: 1,
    explanation: "Tensor ban đầu có 3 trục (rank-3). Việc reshape gộp 2 trục cuối (28, 28) thành một trục duy nhất (784). Shape mới là (10000, 784), do đó nó trở thành Tensor 2 chiều (rank-2)."
  }
];

export default quizData;
