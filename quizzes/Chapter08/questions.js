const questions = [
  {
    type: "mcq",
    question: "Sự khác biệt cơ bản nhất giữa lớp Dense và lớp tích chập (Conv2D) là gì?",
    options: [
      "Lớp Dense học các mẫu cục bộ, trong khi Conv2D học các mẫu chung trong không gian.",
      "Lớp Dense học các mẫu chung toàn cục, trong khi lớp Conv2D học các mẫu cục bộ nhỏ.",
      "Lớp Dense chỉ xử lý hình ảnh màu, còn Conv2D chỉ xử lý hình ảnh đen trắng.",
      "Lớp Dense yêu cầu ít dữ liệu hơn lớp tích chập để học các biểu diễn phân cấp."
    ],
    correctAnswer: 1,
    explanation: "Các lớp Dense tìm hiểu các mẫu chung liên quan đến tất cả các pixel, trong khi các lớp tích chập học các mẫu cục bộ (ví dụ trong các cửa sổ 3x3).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Tại sao ConvNet (CNN) có đặc tính 'bất biến khi dịch' (translation invariance)?",
    options: [
      "Vì nó sử dụng các phép toán phi tuyến tính phức tạp hơn các mạng bình thường.",
      "Vì nó kết hợp với các mô hình ngôn ngữ lớn để hiểu ngữ nghĩa hình ảnh.",
      "Vì sau khi học một mẫu ở một góc, nó có thể nhận ra mẫu đó ở bất kỳ đâu.",
      "Vì nó sử dụng lớp GlobalAveragePooling để loại bỏ hoàn toàn vị trí không gian."
    ],
    correctAnswer: 2,
    explanation: "Vì ConvNet học các mẫu cục bộ, nên sau khi học một mẫu ở một vị trí, nó có thể nhận ra mẫu đó ở vị trí khác, giúp nó hiệu quả với dữ liệu hình ảnh (thế giới thị giác bất biến dịch).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Hệ thống phân cấp không gian (spatial hierarchy) trong ConvNet có nghĩa là gì?",
    options: [
      "Lớp đầu tiên học đặc điểm lớn, các lớp sau chia nhỏ chúng thành pixel.",
      "Tất cả các lớp học cùng một loại đặc điểm ở các độ phân giải khác nhau.",
      "Mô hình phân loại hình ảnh dựa trên vị trí địa lý của đối tượng trong ảnh.",
      "Lớp đầu tiên học cạnh nhỏ, lớp sau học mẫu lớn hơn tạo từ lớp đầu tiên."
    ],
    correctAnswer: 3,
    explanation: "ConvNet học các mô hình phân cấp: lớp đầu tiên học các mẫu cục bộ nhỏ như các cạnh, lớp thứ hai học các mẫu lớn hơn được tạo từ các tính năng của lớp đầu, v.v.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trục 'độ sâu' (depth) của bản đồ đặc trưng (feature map) đầu ra trong một lớp tích chập đại diện cho điều gì?",
    options: [
      "Số lượng các kênh màu RGB (luôn luôn là 3).",
      "Kích thước của cửa sổ tích chập (thường là 3x3).",
      "Số lượng các bộ lọc (filters) được tính toán bởi lớp chập.",
      "Số lượng lớp chập được xếp chồng lên nhau trong toàn bộ mô hình."
    ],
    correctAnswer: 2,
    explanation: "Độ sâu đầu ra là một tham số của lớp và các kênh trong trục độ sâu đó đứng đại diện cho các bộ lọc (filters). Mỗi bộ lọc mã hóa một khía cạnh cụ thể của dữ liệu.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Với lớp `Conv2D(filters=64, kernel_size=3)`, ý nghĩa của 'bản đồ phản hồi' (response map) là gì?",
    options: [
      "Là tập hợp 64 bộ lọc được chia sẻ cho toàn bộ mô hình học sâu.",
      "Là lưới 2D không gian thể hiện phản hồi của một bộ lọc với đầu vào.",
      "Là nhãn phân loại cuối cùng được dự đoán bởi mô hình thần kinh.",
      "Là bản đồ vị trí các pixel bị lỗi trong quá trình thu nhận hình ảnh."
    ],
    correctAnswer: 1,
    explanation: "Mỗi kênh trong đầu ra là một lưới 2D chứa bản đồ phản hồi (response map) của một bộ lọc đối với đầu vào, biểu thị phản hồi của mẫu đó tại các vị trí.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Hai tham số chính xác định cấu trúc của một lớp tích chập là gì?",
    options: [
      "Kích thước cửa sổ (patch size) và số lượng bộ lọc (độ sâu đầu ra).",
      "Kích thước batch size và số lượng bộ lọc (độ sâu đầu ra).",
      "Kích thước cửa sổ (patch size) và hàm kích hoạt (activation function).",
      "Tốc độ học (learning rate) và số lượng epochs khi huấn luyện mô hình."
    ],
    correctAnswer: 0,
    explanation: "Các kết cấu chập được xác định bởi kích thước bản vá (thường là 3x3 hoặc 5x5) và độ sâu của bản đồ tính năng đầu ra (số lượng bộ lọc).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Điều gì xảy ra khi sử dụng `padding='valid'` trong lớp `Conv2D`?",
    options: [
      "Đầu ra sẽ có chiều rộng và chiều cao bằng đúng với đầu vào ban đầu.",
      "Đầu ra sẽ co lại do hiệu ứng đường viền (không có phần đệm được thêm).",
      "Mô hình sẽ tự động bỏ qua các vùng bị nhiễu ở rìa của hình ảnh gốc.",
      "Mô hình sẽ sao chép giá trị pixel ở biên để lấp đầy phần đệm bị thiếu."
    ],
    correctAnswer: 1,
    explanation: "`padding='valid'` có nghĩa là không có phần đệm (no padding), do đó chỉ các vị trí cửa sổ hợp lệ được sử dụng, dẫn đến kích thước bản đồ đặc trưng co lại.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Cấu hình `padding='same'` trong lớp `Conv2D` có tác dụng gì?",
    options: [
      "Tăng gấp đôi kích thước của bản đồ đặc trưng để làm nổi bật chi tiết.",
      "Thêm các hàng và cột bằng 0 để đầu ra có cùng không gian với đầu vào.",
      "Sao chép hoàn toàn cấu trúc của lớp chập trước đó sang lớp hiện tại.",
      "Buộc tất cả các bộ lọc phải có chung một trọng số để giảm chi phí học."
    ],
    correctAnswer: 1,
    explanation: "`padding='same'` có nghĩa là thêm số lượng hàng và cột thích hợp ở các cạnh để thu được bản đồ đặc trưng đầu ra có cùng chiều rộng và chiều cao với đầu vào.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Sải bước (stride) trong một lớp tích chập quyết định điều gì?",
    options: [
      "Số bước tối ưu hóa của thuật toán Adam trong một batch huấn luyện.",
      "Khoảng cách (số pixel) mà cửa sổ tích chập dịch chuyển sau mỗi bước.",
      "Khoảng cách giữa các lớp ẩn trong kiến trúc mạng nơ-ron phân cấp.",
      "Số lượng kênh màu được quét trong một thao tác cuộn (convolution)."
    ],
    correctAnswer: 1,
    explanation: "Bước tiến (stride) là khoảng cách giữa hai cửa sổ liên tiếp. Mặc định là 1, nếu sử dụng stride 2, bản đồ đối tượng sẽ bị giảm kích thước (lấy mẫu xuống) theo hệ số 2.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Tại sao thao tác `MaxPooling2D` thường được ưu tiên sử dụng thay vì Convolution với stride > 1?",
    options: [
      "Vì nó không cần học trọng số, do đó giảm độ phức tạp tính toán đáng kể.",
      "Vì các đối tượng có xu hướng biểu hiện ở 'sự hiện diện tối đa' của mẫu.",
      "Vì nó tăng số lượng kênh đặc trưng lên theo cấp số nhân sau mỗi bước.",
      "Vì nó loại bỏ hoàn toàn nhiễu, giúp các lớp sau dễ nhận diện hơn."
    ],
    correctAnswer: 1,
    explanation: "MaxPooling hoạt động tốt hơn vì sẽ có nhiều thông tin hơn khi xem xét 'sự hiện diện tối đa' của các đối tượng địa lý so với việc xem xét các cửa sổ thưa thớt (qua sải bước) hoặc lấy trung bình.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Lợi ích chính của việc sử dụng các lớp tổng hợp (như `MaxPooling2D`) trong ConvNet là gì?",
    options: [
      "Tăng độ phân giải không gian của hình ảnh để tái tạo lại chi tiết vi mô.",
      "Ngăn cản gradient biến mất trong quá trình lan truyền ngược (backprop).",
      "Tăng tham số để mô hình học các hàm tuyến tính phức tạp dễ dàng hơn.",
      "Giảm kích thước bản đồ đặc trưng, giúp lọc thông tin và phân cấp bộ lọc."
    ],
    correctAnswer: 3,
    explanation: "Việc lấy mẫu xuống giúp giảm kích thước bản đồ đặc trưng, làm thông tin ít phân bố không gian hơn và tạo hệ thống phân cấp bộ lọc để 'nhìn' các cửa sổ lớn hơn so với ảnh gốc.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Đối tượng `tf.data.Dataset` trong TensorFlow đóng vai trò gì?",
    options: [
      "Một trình tối ưu hóa giúp giảm hàm mất mát nhanh hơn thuật toán Adam.",
      "Một vòng lặp huấn luyện tùy chỉnh hoàn toàn thay thế cho phương thức fit().",
      "Một API tạo đường ống đầu vào hiệu quả, xử lý batching và prefetching.",
      "Một kho lưu trữ các mô hình được đào tạo trước giống như HuggingFace."
    ],
    correctAnswer: 2,
    explanation: "Dataset API tạo đường ống đầu vào hiệu quả, hỗ trợ lặp, tải dữ liệu, batching, parallelization preprocessing và asynchronous data prefetching.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Phương thức `.prefetch()` của đối tượng Dataset mang lại lợi ích hiệu suất nhờ cơ chế nào?",
    options: [
      "Xóa bỏ các hình ảnh bị hỏng trước khi đưa chúng vào pipeline huấn luyện.",
      "Nén kích thước của batch để tiết kiệm dung lượng RAM trên hệ thống.",
      "Tự động tăng cường dữ liệu dựa trên phản hồi của quá trình huấn luyện.",
      "Xử lý trước lô dữ liệu tiếp theo trong khi mô hình đang chạy lô hiện tại."
    ],
    correctAnswer: 3,
    explanation: "Prefetching cho phép chuẩn bị lô dữ liệu tiếp theo trong bộ nhớ thiết bị song song với việc mô hình thực thi trên lô trước đó, tránh tắc nghẽn luồng.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Mục đích của kỹ thuật Tăng cường dữ liệu (Data Augmentation) trong thị giác máy tính là gì?",
    options: [
      "Tạo thêm hình ảnh bằng cách biến đổi ngẫu nhiên nhằm giảm overfitting.",
      "Tăng độ phân giải của hình ảnh gốc để mô hình học được chi tiết nhỏ.",
      "Tự động tạo nhãn cho dữ liệu chưa được gán nhãn bằng học bán giám sát.",
      "Tăng tốc độ hội tụ của mô hình bằng cách loại bỏ các pixel không cần thiết."
    ],
    correctAnswer: 0,
    explanation: "Data Augmentation tạo ra nhiều dữ liệu huấn luyện hơn bằng cách tăng cường các mẫu thông qua các phép biến đổi ngẫu nhiên, giúp mô hình không thấy cùng một ảnh hai lần, từ đó chống quá khớp.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Theo Keras, cách tiếp cận tốt hơn để áp dụng Data Augmentation bằng các lớp `RandomFlip`, `RandomRotation` là gì?",
    options: [
      "Áp dụng vào cuối mạng nơ-ron, ngay trước lớp Softmax phân loại.",
      "Áp dụng trong đường ống dữ liệu (data pipeline) qua lệnh map() ở CPU.",
      "Áp dụng vào tập dữ liệu thử nghiệm (test set) để đo lường tính ổn định.",
      "Áp dụng trực tiếp vào trọng số của mô hình sau mỗi epoch huấn luyện."
    ],
    correctAnswer: 1,
    explanation: "Đưa data augmentation vào pipeline dữ liệu thông qua dataset.map() chạy trên CPU nhiều lõi song song, giải phóng GPU để chỉ tập trung vào việc huấn luyện mô hình.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Tăng cường dữ liệu (Data Augmentation) thường CHỈ áp dụng ở giai đoạn nào?",
    options: [
      "Ở giai đoạn đánh giá mô hình (evaluate).",
      "Ở giai đoạn dự đoán trên tập Test (predict).",
      "Ở giai đoạn huấn luyện mô hình (training).",
      "Ở giai đoạn biên dịch mô hình (compile)."
    ],
    correctAnswer: 2,
    explanation: "Data augmentation là kỹ thuật điều chuẩn hóa (regularization), do đó nó chỉ được áp dụng trong thời gian huấn luyện. Khi đánh giá (test/validation), chúng ta sử dụng hình ảnh chưa được tăng cường.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Mô hình được huấn luyện trước (Pretrained model) mang lại lợi ích chính nào?",
    options: [
      "Luôn luôn cho độ chính xác 100% không phụ thuộc vào dữ liệu đầu vào.",
      "Tái sử dụng các đặc trưng phân cấp đã học trên tập dữ liệu lớn và chung.",
      "Giảm dung lượng file mô hình xuống mức tối thiểu nhờ kỹ thuật nén.",
      "Chỉ sử dụng được cho những tập dữ liệu có cùng số lớp với mô hình gốc."
    ],
    correctAnswer: 1,
    explanation: "Mô hình pre-trained học được các đặc trưng từ tập dữ liệu lớn. Phân cấp không gian này rất tổng quát và có thể tái sử dụng làm bộ trích xuất đặc trưng cho nhiều bài toán mới.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong kiến trúc ConvNet, phần nào được gọi là 'Cơ sở tích chập' (convolutional base/backbone)?",
    options: [
      "Phần lớp phân loại (Dense) cuối cùng xử lý đầu ra mạng nơ-ron.",
      "Chuỗi các lớp pooling tích chập và gộp ở giai đoạn đầu của mô hình.",
      "Hàm tính toán sự mất mát (loss) và thuật toán tối ưu hóa của mô hình.",
      "Thành phần dữ liệu đầu vào (Input layer) và bước tiền xử lý (Scaling)."
    ],
    correctAnswer: 1,
    explanation: "Cơ sở tích chập (backbone) bao gồm loạt các lớp convolution và pooling ở phần đầu mô hình, làm nhiệm vụ trích xuất bản đồ tính năng (feature extraction).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Khi thực hiện Feature Extraction, tại sao ta lại loại bỏ bộ phân loại (classifier/Dense layers) cũ của mô hình Pretrained?",
    options: [
      "Vì các lớp Dense không tương thích với phiên bản Keras hiện tại.",
      "Vì các lớp Dense chứa biểu diễn đặc trưng chung (generic representations).",
      "Vì biểu diễn ở lớp Dense quá đặc thù với tập các lớp của bài toán gốc.",
      "Vì việc sử dụng các lớp Dense sẽ gây ra tình trạng bùng nổ gradient."
    ],
    correctAnswer: 2,
    explanation: "Biểu diễn ở cơ sở tích chập có tính chung (cạnh, màu, kết cấu), trong khi các lớp Dense hoàn toàn loại bỏ không gian và biểu diễn xác suất rất đặc thù cho các nhãn phân loại cũ.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Việc 'đóng băng' (freezing) cơ sở tích chập trong Keras thực hiện bằng cách nào?",
    options: [
      "Gọi hàm `model.freeze()` trước khi thiết lập trình biên dịch compile.",
      "Thiết lập thuộc tính `trainable = False` cho mô hình hoặc lớp tích chập.",
      "Đặt đối số `freezing=True` trong hàm `model.fit()` khi huấn luyện mạng.",
      "Sử dụng một Callback đặc biệt tên là `FreezeLayers` trong khi chạy fit."
    ],
    correctAnswer: 1,
    explanation: "Trong Keras, đóng băng mạng đồng nghĩa với việc cản trở cập nhật trọng số trong khi huấn luyện, thực hiện thông qua việc gán `trainable = False`.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Tại sao BẮT BUỘC phải đóng băng cơ sở tích chập (freezing) khi ghép nối với một bộ phân loại (classifier) mới chưa được huấn luyện?",
    options: [
      "Để bộ nhớ RAM không bị quá tải khi tính toán trên dữ liệu hình ảnh lớn.",
      "Để giảm thời gian huấn luyện mô hình (epochs diễn ra nhanh chóng hơn).",
      "Để tránh gradient lớn phá hủy các đặc trưng đã học của cơ sở tích chập.",
      "Để đảm bảo tốc độ học (learning rate) tự động được giảm theo cấp số."
    ],
    correctAnswer: 2,
    explanation: "Vì các lớp Dense mới được khởi tạo ngẫu nhiên, việc cập nhật trọng số sẽ cực lớn lúc ban đầu. Nếu không đóng băng, sự thay đổi này sẽ lan truyền và phá hủy toàn bộ các đặc trưng tốt đã học trước đó ở backbone.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Kỹ thuật 'Tinh chỉnh' (Fine-tuning) bao gồm quá trình nào?",
    options: [
      "Mở băng các lớp trên cùng của mạng cơ sở và huấn luyện cùng bộ phân loại.",
      "Tăng kích thước batch size lên mức tối đa để khai thác tối ưu GPU.",
      "Thay thế toàn bộ kiến trúc mô hình bằng một mạng phức tạp như ResNet.",
      "Mở băng các lớp dưới cùng (đầu vào) để học các mẫu cạnh cơ bản lại."
    ],
    correctAnswer: 0,
    explanation: "Tinh chỉnh (Fine-tuning) bao gồm việc mở băng (unfreeze) một phần mô hình cơ sở đã đóng băng, đặc biệt là các lớp trên cùng (top layers), và huấn luyện đồng thời nó cùng với bộ phân loại mới.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trong quá trình Fine-tuning, tại sao ta lại ưu tiên tinh chỉnh các lớp trên cùng (top layers) của cơ sở tích chập thay vì các lớp đầu tiên?",
    options: [
      "Vì các lớp đầu tiên có số lượng tham số lớn nhất, gây tốn kém thời gian.",
      "Vì các lớp đầu tiên mã hóa đặc trưng rất chung, còn lớp trên mã hóa khái niệm đặc thù.",
      "Vì các lớp đầu tiên không thể cập nhật gradient theo nguyên lý lan truyền.",
      "Vì hàm mất mát không bao giờ tác động được xuống các lớp ở phần đầu."
    ],
    correctAnswer: 1,
    explanation: "Các lớp ban đầu mã hóa các đặc điểm chung, tái sử dụng cao. Các lớp trên mã hóa các tính năng chuyên biệt (mắt chó, tai mèo). Tinh chỉnh cần đánh vào lớp chuyên biệt để định hướng lại cho bài toán mới.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Điều kiện tiên quyết trước khi tiến hành bước Fine-tuning là gì?",
    options: [
      "Phải tăng kích thước ảnh đầu vào (input size) gấp đôi để phân tích kỹ.",
      "Bộ phân loại (Dense) trên cùng phải được huấn luyện cho hội tụ trước.",
      "Phải xóa toàn bộ lớp Dropout ra khỏi mô hình để tránh thất thoát thông tin.",
      "Phải sử dụng bộ tối ưu hóa SGD thay vì Adam để tinh chỉnh chính xác."
    ],
    correctAnswer: 1,
    explanation: "Chỉ được phép Fine-tuning cơ sở chập sau khi bộ phân loại (classifier) mới được thêm vào đã được huấn luyện đầy đủ. Nếu chưa huấn luyện, lỗi lớn sẽ phá hủy trọng số.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Yêu cầu bắt buộc về 'tốc độ học' (learning rate) khi thực hiện Fine-tuning là gì?",
    options: [
      "Sử dụng tốc độ học rất cao (ví dụ 0.1) để vượt qua cực tiểu cục bộ.",
      "Giữ nguyên tốc độ học như lúc huấn luyện bộ phân loại mới khởi tạo.",
      "Tốc độ học phải thay đổi ngẫu nhiên theo từng epoch để tối ưu hóa.",
      "Sử dụng tốc độ học rất nhỏ (như 1e-5) để hạn chế phá vỡ biểu diễn cũ."
    ],
    correctAnswer: 3,
    explanation: "Việc cập nhật với một tốc độ học nhỏ là cần thiết để giới hạn độ lớn của các điều chỉnh lên trọng số của lớp được fine-tune, bảo vệ các biểu diễn đặc trưng không bị phá hỏng.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Ghép nối các kỹ thuật xử lý dữ liệu nhỏ với mô tả tương ứng:",
    pairs: [
      { left: "Train from scratch", right: "Tự học toàn bộ trọng số mạng từ dữ liệu ban đầu, dễ bị quá khớp" },
      { left: "Data Augmentation", right: "Biến đổi ngẫu nhiên ảnh huấn luyện (xoay, lật) để tạo tính đa dạng" },
      { left: "Feature Extraction", right: "Tận dụng cấu trúc Convolutional Base đã học để sinh đặc trưng" },
      { left: "Fine-tuning", right: "Mở băng lớp đỉnh của Backbone để tinh chỉnh nhẹ cho bài toán mới" }
    ],
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Ghép nối các thông số của Convolutional Network với ý nghĩa của nó:",
    pairs: [
      { left: "Patch Size (Kernel)", right: "Kích thước của cửa sổ quét qua bản đồ (thường 3x3)" },
      { left: "Output Depth (Filters)", right: "Số lượng bộ lọc được áp dụng, tạo ra số kênh tương ứng" },
      { left: "Padding", right: "Đệm viền (bằng 0) để xử lý hiệu ứng biên của bức ảnh" },
      { left: "Stride", right: "Bước nhảy của cửa sổ trượt, quyết định mức độ giảm chiều" }
    ],
    difficulty: "Dễ"
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự các bước đúng khi thực hiện kỹ thuật Fine-Tuning một mô hình huấn luyện trước:",
    steps: [
      "Thêm mạng tùy chỉnh (Dense classifier) lên trên mạng cơ sở (Base network).",
      "Đóng băng toàn bộ mạng cơ sở (Base network).",
      "Huấn luyện bộ phân loại mới (Dense classifier) cho hội tụ.",
      "Mở băng mạng cơ sở (hoặc vài lớp đỉnh của nó).",
      "Huấn luyện chung bộ phân loại và các lớp mở băng với tốc độ học cực nhỏ."
    ],
    explanation: "Đầu tiên thêm bộ phân loại, sau đó phải đóng băng backbone và huấn luyện bộ phân loại trước. Khi đã ổn định, mới mở băng backbone (hoặc phần đỉnh) và train lại với learning_rate nhỏ.",
    difficulty: "Khó"
  },
  {
    type: "fill",
    question: "Một hoạt động trích xuất các bản vá từ đầu vào cục bộ và tính toán giá trị 'tối đa' trên từng kênh, giúp giảm mạnh độ phân giải bản đồ đặc trưng, được gọi là hoạt động {1}.",
    blanks: [
      { id: 1, text: "tổng hợp tối đa", answer: "max pooling" }
    ],
    explanation: "Max pooling (Tổng hợp tối đa) có chức năng downsampling bản đồ tính năng bằng việc trích xuất giá trị tối đa trong cửa sổ (thường 2x2).",
    difficulty: "Dễ"
  },
  {
    type: "fill",
    question: "Việc giữ cho trọng số của một lớp không bị thay đổi (cập nhật) trong quá trình lan truyền ngược (backpropagation) được Keras gọi là {1} (thông qua thuộc tính trainable = False).",
    blanks: [
      { id: 1, text: "đóng băng", answer: "freezing" }
    ],
    explanation: "Freezing (Đóng băng) là ngăn không cho các trọng số được cập nhật trong lúc huấn luyện mạng.",
    difficulty: "Trung bình"
  }
];

export default questions;
