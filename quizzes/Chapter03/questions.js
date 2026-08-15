const quizData = [
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 1: Thư viện Keras được phát hành lần đầu tiên vào thời gian nào?",
    options: [
      "Tháng 3 năm 2013",
      "Tháng 3 năm 2015",
      "Tháng 11 năm 2015",
      "Tháng 9 năm 2016"
    ],
    correctAnswer: 1,
    explanation: "Keras là framework lâu đời nhất trong số bộ tứ (Keras, TensorFlow, PyTorch, JAX), được Francois Chollet phát hành vào tháng 3 năm 2015."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 2: Ba tính năng chính mà MỌI framework deep learning hiện đại đều có là gì?",
    options: [
      "Xây dựng giao diện web, quản lý cơ sở dữ liệu và xử lý video.",
      "Lập lịch tác vụ, tính toán ma trận thưa và triển khai đám mây.",
      "Tự động vi phân, tính toán tensor trên GPU/TPU, và phân tán tính toán.",
      "Xử lý ngôn ngữ tự nhiên, xử lý ảnh y tế và điều khiển robot."
    ],
    correctAnswer: 2,
    explanation: "Tất cả framework học sâu đều kết hợp 3 thứ: (1) Tính đạo hàm tự động, (2) Tính toán tensor trên phần cứng chuyên dụng (GPU/TPU), (3) Phân tán tính toán trên nhiều thiết bị."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 3: Đâu là framework đóng vai trò là 'tổ tiên' khái niệm cho tất cả các công cụ học sâu hiện đại (ra đời khoảng năm 2009)?",
    options: [
      "Theano",
      "TensorFlow",
      "PyTorch",
      "Caffe"
    ],
    correctAnswer: 0,
    explanation: "Theano (2009) là framework đầu tiên hỗ trợ vi phân tự động và tính toán GPU để huấn luyện mạng nơ-ron."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 4: Google ra mắt TensorFlow vào cuối năm 2015 với cải tiến lớn nào so với Theano?",
    options: [
      "Loại bỏ hoàn toàn ngôn ngữ Python để chuyển sang dùng C++.",
      "Hỗ trợ phân tán tính toán quy mô lớn trên nhiều máy chủ.",
      "Chỉ hỗ trợ chạy mô hình trên trình duyệt web và di động.",
      "Tích hợp sẵn các mô hình ngôn ngữ lớn như GPT."
    ],
    correctAnswer: 1,
    explanation: "TensorFlow lấy ý tưởng từ Theano nhưng thêm khả năng tính toán phân tán trên quy mô rất lớn."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 5: Meta (Facebook) ra mắt PyTorch vào năm 2016 nhằm đối trọng với TensorFlow, lấy cảm hứng từ đâu?",
    options: [
      "Chainer và thư viện NumPy-Autograd",
      "Theano và thư viện Caffe của Berkeley",
      "Keras và thư viện MXNet của Amazon",
      "C++ Standard Library và thư viện Torch 7"
    ],
    correctAnswer: 0,
    explanation: "PyTorch lấy ý tưởng thiết kế lập trình linh hoạt từ Chainer (Nhật Bản) và NumPy-Autograd."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 6: Framework học sâu cực nhanh do Matthew Johnson (Google) phát triển dựa trên XLA được gọi là ________.",
    blanks: ["JAX", "jax"],
    explanation: "JAX là một framework rất phổ biến trong giới nghiên cứu, kết hợp Autograd và trình biên dịch XLA."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 7: So với TensorFlow, PyTorch và JAX, Keras khác biệt ở điểm nào?",
    options: [
      "Keras không hỗ trợ tính toán tensor trên GPU.",
      "Keras là framework cấp cao (high-level), giống như bộ lắp ráp đúc sẵn.",
      "Keras là framework cấp thấp (low-level), yêu cầu tự viết Backpropagation.",
      "Keras chỉ hỗ trợ xử lý ngôn ngữ tự nhiên (NLP)."
    ],
    correctAnswer: 1,
    explanation: "Nếu TensorFlow/PyTorch/JAX là vật liệu xây dựng (gạch, xi măng), thì Keras là bộ nhà lắp ghép cung cấp các thành phần cấp cao (Layer, Optimizer, Model)."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 8: Keras có thể chạy trên NumPy (NumPy backend), nhưng bị hạn chế lớn nhất là gì?",
    options: [
      "Không thể chạy trên CPU của máy tính.",
      "Chỉ giới hạn xử lý ảnh đen trắng kích thước nhỏ.",
      "Không thể huấn luyện mô hình vì NumPy không hỗ trợ đạo hàm tự động.",
      "Không tương thích với các phiên bản Python mới."
    ],
    correctAnswer: 2,
    explanation: "Vì NumPy không có API tính gradient (đạo hàm), backend NumPy trong Keras chỉ dùng để dự đoán (inference) chứ không thể dùng để huấn luyện (train)."
  },
  {
    type: "sorting",
    difficulty: "Trung bình",
    question: "Câu 9: Sắp xếp các framework học sâu theo thứ tự thời gian phát hành lần đầu tiên (từ cũ nhất đến mới nhất):",
    steps: [
      "Theano",
      "Keras",
      "TensorFlow",
      "PyTorch",
      "JAX"
    ],
    explanation: "Thứ tự ra mắt: Theano (2009) -> Keras (Tháng 3/2015) -> TensorFlow (Tháng 11/2015) -> PyTorch (2016) -> JAX (2018)."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 10: Trong TensorFlow, cách tạo một tensor toàn số 0 (zeros tensor) có kích thước (2, 2) là gì?",
    options: [
      "tf.array([[0, 0], [0, 0]])",
      "tf.zeros(shape=(2, 2))",
      "tf.fill((2, 2), 0)",
      "tf.empty(shape=(2, 2))"
    ],
    correctAnswer: 1,
    explanation: "Hàm `tf.zeros(shape=(2, 2))` dùng để tạo tensor toàn số 0, tương tự như `np.zeros`."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 11: Sự khác biệt cốt lõi giữa tensor trong NumPy (ndarray) và tensor trong TensorFlow (tf.Tensor) là gì?",
    options: [
      "Tensor của TensorFlow không hỗ trợ kiểu số thực float32.",
      "Tensor của NumPy có thể chạy trực tiếp trên card đồ họa (GPU).",
      "Tensor của TensorFlow là hằng số (constant) và không thể gán lại giá trị mới.",
      "Tensor của TensorFlow chỉ dùng được cho mảng tối đa 2 chiều."
    ],
    correctAnswer: 2,
    explanation: "Trong TensorFlow, bạn không thể thay đổi giá trị một phần tử của `tf.Tensor` (như `x[0,0] = 1`). Để lưu trữ trạng thái có thể thay đổi (ví dụ: trọng số), bạn phải dùng `tf.Variable`."
  },
  {
    type: "fill_blank",
    difficulty: "Dễ",
    question: "Câu 12: Để lưu trữ và cập nhật trọng số mạng lưới trong TensorFlow, ta phải dùng lớp ________ thay vì tensor thông thường.",
    blanks: ["tf.Variable", "Variable", "variable"],
    explanation: "`tf.Variable` được thiết kế để chứa các giá trị có thể cập nhật (thông qua hàm `.assign()`), rất cần thiết để lưu trữ weights và biases."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 13: Trong TensorFlow, làm cách nào để tính đạo hàm (gradient) của một biểu thức toán học?",
    options: [
      "Sử dụng hàm tf.math.derivative()",
      "Mở phạm vi tf.GradientTape() để theo dõi các phép toán",
      "Gọi phương thức .backward() trực tiếp trên tensor kết quả",
      "Biên dịch mô hình với cờ compute_gradient=True"
    ],
    correctAnswer: 1,
    explanation: "TensorFlow dùng `tf.GradientTape` để ghi lại lịch sử các phép toán vào 'băng từ', sau đó gọi `tape.gradient(loss, weights)` để lấy đạo hàm."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 14: Theo mặc định, tf.GradientTape theo dõi (watch) những tensor nào?",
    options: [
      "Tất cả các biến (tf.Variable) có thể huấn luyện được (trainable=True).",
      "Tất cả mọi tensor được khai báo bên trong phạm vi with của nó.",
      "Chỉ những tensor là hằng số (tf.constant).",
      "Bất kỳ tensor nào được đưa vào làm đầu vào của hàm mạng nơ-ron."
    ],
    correctAnswer: 0,
    explanation: "Để tiết kiệm bộ nhớ, GradientTape tự động theo dõi `tf.Variable` (trainable). Nếu muốn tính đạo hàm theo một `tf.constant`, bạn phải gọi `tape.watch()` thủ công."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 15: Chế độ thực thi mặc định của TensorFlow (Eager Execution) có đặc điểm gì?",
    options: [
      "Các lệnh được biên dịch trước thành đồ thị rồi mới chạy.",
      "Code chạy rất nhanh nhưng không thể debug bằng lệnh print().",
      "Các phép toán chạy từng dòng một giống hệt code Python thông thường.",
      "Chỉ tương thích với phần cứng TPU của Google."
    ],
    correctAnswer: 2,
    explanation: "Eager execution (thực thi hăng hái) chạy code từng dòng ngay lập tức. Nó rất dễ debug nhưng tốc độ chạy chậm hơn đồ thị biên dịch."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 16: Để tăng tốc mã TensorFlow, ta dùng trình biên dịch (compiler) tích hợp bằng cách thêm decorator nào?",
    options: [
      "@tf.compile",
      "@tf.optimize",
      "@tf.function",
      "@tf.fast_mode"
    ],
    correctAnswer: 2,
    explanation: "Thêm `@tf.function` trước một hàm Python sẽ chuyển nó thành đồ thị TensorFlow tĩnh (graph mode), tối ưu hóa các phép toán và chạy cực nhanh."
  },
  {
    type: "matching",
    difficulty: "Dễ",
    question: "Câu 17: Ghép nối tên framework với công ty/tổ chức phát triển chính:",
    pairs: [
      { left: "TensorFlow", right: "Google" },
      { left: "PyTorch", right: "Meta (Facebook)" },
      { left: "JAX", right: "Google Research" },
      { left: "Keras", right: "Francois Chollet & Open Source" }
    ],
    explanation: "TensorFlow và JAX thuộc hệ sinh thái Google. PyTorch do phòng nghiên cứu AI của Meta (FAIR) phát triển. Keras độc lập nhưng gắn chặt với TensorFlow."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 18: Trình biên dịch XLA (Accelerated Linear Algebra) có thể được kích hoạt trong TensorFlow bằng tham số nào?",
    options: [
      "@tf.function(xla_mode=True)",
      "@tf.function(jit_compile=True)",
      "@tf.function(optimize='XLA')",
      "@tf.function(fast_math=True)"
    ],
    correctAnswer: 1,
    explanation: "Kích hoạt JIT compilation (Just-In-Time) bằng XLA thông qua `@tf.function(jit_compile=True)` giúp gộp (fuse) các phép toán lại, tăng tốc độ đáng kể."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 19: Trong PyTorch, để tạo một tensor ngẫu nhiên theo phân phối đều (uniform) có kích thước (3, 1), cú pháp nào ĐÚNG?",
    options: [
      "torch.random.uniform(shape=(3, 1))",
      "torch.rand(3, 1)",
      "torch.uniform((3, 1))",
      "torch.random(size=(3, 1))"
    ],
    correctAnswer: 1,
    explanation: "Khác với NumPy hay TensorFlow dùng tuple `(3, 1)`, hàm `torch.rand` trong PyTorch nhận kích thước chiều trực tiếp làm các đối số truyền vào: `torch.rand(3, 1)`."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 20: Tên thư viện để import PyTorch vào mã nguồn Python là gì?",
    options: [
      "import pytorch",
      "import pt",
      "import torch",
      "import pyt"
    ],
    correctAnswer: 2,
    explanation: "Mặc dù framework tên là PyTorch, gói cài đặt (package) trong Python luôn được gọi tắt là `torch`."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 21: PyTorch khác biệt với TensorFlow ở điểm nào về việc gán giá trị cho Tensor (Tensor assignment)?",
    options: [
      "Tensor của PyTorch không thể thay đổi kích thước.",
      "Tensor của PyTorch không thể cộng với Tensor của TensorFlow.",
      "Tensor của PyTorch CÓ THỂ gán lại giá trị trực tiếp (như x[0,0] = 1).",
      "Tensor của PyTorch yêu cầu trình biên dịch riêng."
    ],
    correctAnswer: 2,
    explanation: "Không giống như `tf.Tensor` bị khóa giá trị (immutable), `torch.Tensor` cho phép thay đổi giá trị cục bộ giống hệt như mảng NumPy."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 22: Phương thức dùng để tính đạo hàm ngược (backpropagation) tự động trên một tensor trong PyTorch là ________.",
    blanks: ["backward()", "backward"],
    explanation: "Trong PyTorch, sau khi tính toán ra giá trị `loss`, ta chỉ cần gọi `loss.backward()` là tự động tính toàn bộ gradient cho các tensor."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 23: Lỗi 'EagerTensor object does not support item assignment' xảy ra khi bạn cố gắng làm gì?",
    options: [
      "Gọi hàm backward() hai lần liên tiếp trong một vòng lặp huấn luyện.",
      "Cố gắng sửa đổi giá trị của một tf.Tensor bằng cú pháp chỉ mục (ví dụ: x[0]=1).",
      "Sử dụng trình tối ưu hóa không tương thích với Keras.",
      "Chạy một hàm được bọc bởi @tf.function với các tham số khác nhau."
    ],
    correctAnswer: 1,
    explanation: "Lỗi này ở TensorFlow báo hiệu bạn đang lấy 1 tensor hằng số và cố thay đổi nó. Bạn phải dùng `tf.Variable` và phương thức `.assign()` thay thế."
  },
  {
    type: "sorting",
    difficulty: "Trung bình",
    question: "Câu 24: Sắp xếp các bước chuẩn của vòng lặp huấn luyện trong PyTorch:",
    steps: [
      "Đưa dữ liệu qua mô hình (Forward pass)",
      "Tính toán hàm mất mát (Compute loss)",
      "Xóa sạch gradient cũ bằng optimizer.zero_grad()",
      "Tính gradient ngược bằng loss.backward()",
      "Cập nhật trọng số bằng optimizer.step()"
    ],
    explanation: "Quy trình chuẩn PyTorch: Forward -> Loss -> Zero Grad (cực kỳ quan trọng) -> Backward -> Step."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 25: Khi thiết kế một lớp (layer) tùy chỉnh trong PyTorch, ta cần kế thừa từ class nào?",
    options: [
      "torch.Layer",
      "torch.nn.Module",
      "torch.models.Base",
      "torch.Network"
    ],
    correctAnswer: 1,
    explanation: "Tất cả các lớp và mô hình trong PyTorch đều phải kế thừa từ lớp cơ sở `torch.nn.Module`."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 26: Khuyết điểm lớn nhất khi mã TensorFlow chạy ở chế độ đồ thị biên dịch (Graph Mode) là gì?",
    options: [
      "Nó làm tiêu hao pin nhanh hơn trên thiết bị di động.",
      "Mã chạy rất chậm so với Eager execution.",
      "Rất khó để gỡ lỗi (debugging) vì mã thực thi không còn là mã Python ban đầu.",
      "Không hỗ trợ huấn luyện trên nhiều GPU cùng lúc."
    ],
    correctAnswer: 2,
    explanation: "Khi mã được biên dịch bằng `@tf.function`, nó được tách khỏi Python. Các hàm `print()` sẽ không in ra giá trị biến khi chạy, làm việc gỡ lỗi rất khó khăn."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 27: Điểm chung của NumPy-Autograd và JAX là gì?",
    options: [
      "Cả hai đều do Meta (Facebook) phát triển độc quyền.",
      "Cả hai đều không thể tính gradient bậc hai (second-order gradients).",
      "Cả hai đều chỉ dùng API của NumPy nhưng mở rộng thêm tính năng tự động vi phân.",
      "Cả hai đều đã bị khai tử và không còn ai sử dụng."
    ],
    correctAnswer: 2,
    explanation: "JAX được phát triển bởi Matthew Johnson (người từng tạo ra NumPy-Autograd). API của JAX là bản sao chính xác của NumPy (`import jax.numpy as jnp`) cộng thêm vi phân và chạy GPU."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 28: Keras được ví von với hình ảnh gì so với TensorFlow/PyTorch?",
    options: [
      "Vật liệu xây thô (gạch, xi măng).",
      "Bộ công cụ lắp ghép nhà đúc sẵn (prefabricated building kit).",
      "Một hệ điều hành quản lý phần cứng.",
      "Một chiếc máy ủi dọn dẹp dữ liệu."
    ],
    correctAnswer: 1,
    explanation: "Nếu TF/PyTorch là vật liệu thô (tensor, đạo hàm), thì Keras là bộ lắp ghép thông minh (Layers, Model, fit) giúp xây dựng mạng nơ-ron nhanh chóng."
  },
  {
    type: "fill_blank",
    difficulty: "Dễ",
    question: "Câu 29: Theo chuẩn Keras/TensorFlow, chiều của các kênh màu (color channels) nằm ở vị trí nào trong hình dạng của ảnh tensor?",
    blanks: ["cuối cùng", "cuối", "last"],
    explanation: "Keras dùng chuẩn channels-last, ví dụ: (samples, height, width, channels). Ngược lại, PyTorch dùng channels-first."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 30: Phương thức nào hiệu quả nhất để cộng thêm giá trị vào một tf.Variable trong TensorFlow?",
    options: [
      "v = v + 1",
      "v.add(1)",
      "v.assign_add(1)",
      "tf.math.add(v, 1)"
    ],
    correctAnswer: 2,
    explanation: "Dùng phương thức `.assign_add()` sẽ cập nhật giá trị biến trực tiếp tại chỗ (in-place), đây là cách chuẩn xác và tối ưu nhất trong TensorFlow."
  }
];

export default quizData;
