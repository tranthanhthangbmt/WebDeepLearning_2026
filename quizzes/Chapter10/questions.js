const questions = [
  {
    type: "mcq",
    question: "Vấn đề cơ bản nào mà chúng ta cần giải quyết khi xây dựng ứng dụng thị giác máy tính cho các lĩnh vực như y tế (medical imaging)?",
    options: [
      "Tối ưu hóa dung lượng bộ nhớ lưu trữ để mô hình có thể chạy được trên mọi thiết bị.",
      "Cải thiện khả năng diễn giải (interpretability) để hiểu tại sao mạng đưa ra quyết định.",
      "Tăng tốc độ dự đoán của mô hình lên mức thời gian thực trên các hệ thống CPU.",
      "Bổ sung càng nhiều dữ liệu giả mạo (augmentation) càng tốt để đánh lừa mô hình."
    ],
    correctAnswer: 1,
    explanation: "Trong các lĩnh vực cần sự tham gia của chuyên gia (như y tế), việc hiểu *tại sao* mô hình lại đưa ra chẩn đoán đó (interpretability) là vô cùng quan trọng để tạo niềm tin và kiểm chứng.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Nhận định nào sau đây là ĐÚNG về khả năng diễn giải (interpretability) của mạng ConvNets so với các mô hình học sâu khác?",
    options: [
      "ConvNets hoàn toàn là hộp đen, không có bất kỳ cách nào để con người có thể hiểu.",
      "ConvNets có khả năng diễn giải rất kém vì nó chỉ dựa trên toán học ma trận 1D.",
      "ConvNets có khả năng trực quan hóa cao vì nó học các biểu diễn khái niệm hình ảnh.",
      "ConvNets yêu cầu người dùng phải tự tính toán lại gradient bằng tay để kiểm tra."
    ],
    correctAnswer: 2,
    explanation: "Không giống như một số mô hình deep learning khác là 'hộp đen', các biểu diễn học được của ConvNets có khả năng trực quan hóa cao vì chúng là các đặc trưng đồ họa không gian 2D.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Trực quan hóa kích hoạt trung gian (intermediate activations) hữu ích nhất cho mục đích gì?",
    options: [
      "Hiểu cách các lớp tích chập liên tiếp biến đổi một đầu vào cụ thể như thế nào.",
      "Tìm ra giá trị siêu tham số tốc độ học (learning rate) tối ưu cho toàn mô hình.",
      "Đánh giá chính xác 100% thời gian thực thi của mỗi lớp mạng lưới khi tính toán.",
      "Loại bỏ các dữ liệu rác (noise) có trong bộ dữ liệu hình ảnh đào tạo gốc."
    ],
    correctAnswer: 0,
    explanation: "Kích hoạt trung gian hiển thị bản đồ đặc trưng (feature maps) sau mỗi lớp tích chập, giúp chúng ta thấy rõ mạng lưới đang chú ý đến đặc điểm gì của bức ảnh tại tầng đó.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Khi trực quan hóa bản đồ đặc trưng của một lớp tích chập, cách hiển thị đúng nhất là gì?",
    options: [
      "Tính trung bình tất cả các kênh (channels) rồi hiển thị dưới dạng một ảnh đen trắng.",
      "Chỉ hiển thị kênh đầu tiên vì nó chứa toàn bộ thông tin quan trọng nhất của bức ảnh.",
      "Hiển thị độc lập từng kênh của lớp đó dưới dạng các hình ảnh 2D riêng biệt.",
      "Cộng gộp kênh đỏ, lục, lam lại để tạo ra một hình ảnh màu sắc rực rỡ nhất."
    ],
    correctAnswer: 2,
    explanation: "Vì mỗi kênh (channel) trong một lớp Conv2D học một đặc trưng độc lập (như cạnh ngang, mắt mèo...), cách tốt nhất là vẽ độc lập nội dung mỗi kênh dưới dạng một ảnh 2D.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Để trích xuất các kích hoạt trung gian (activations), chúng ta cần tạo ra một mô hình Keras có đặc điểm gì đặc biệt?",
    options: [
      "Mô hình không cần bất kỳ đầu ra (output) nào, chỉ nhận đầu vào.",
      "Mô hình có nhiều đầu vào (multi-input) nhưng chỉ có một đầu ra.",
      "Mô hình có cùng đầu vào nhưng có rất nhiều đầu ra (multi-output).",
      "Mô hình đảo ngược chiều tính toán từ đầu ra về lại ảnh đầu vào."
    ],
    correctAnswer: 2,
    explanation: "Ta tạo một mô hình Keras nhận đầu vào là ảnh gốc, và có đầu ra (outputs) là một danh sách chứa tất cả các output của các lớp Conv2D và MaxPooling trung gian.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Theo quan sát khi trực quan hóa, các lớp tích chập ĐẦU TIÊN (gần đầu vào) thường học những đặc trưng gì?",
    options: [
      "Các khái niệm cực kỳ trừu tượng như hình dáng toàn thân của con mèo.",
      "Các bộ phát hiện cạnh (edge detectors), góc và các đốm màu đơn giản.",
      "Chỉ mã hóa các nhãn phân loại bằng các ma trận số 0 và số 1 rời rạc.",
      "Gần như không học được gì vì mạng chưa đủ độ sâu để tính toán lỗi."
    ],
    correctAnswer: 1,
    explanation: "Các lớp đầu tiên hoạt động như một tập hợp các bộ phát hiện cạnh đơn giản và giữ lại gần như toàn bộ chi tiết vật lý của bức ảnh gốc.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Khi tiến sâu hơn vào mạng ConvNet, đặc điểm của các bản đồ kích hoạt (activation maps) thay đổi như thế nào?",
    options: [
      "Trở nên ngày càng rõ nét hơn và khôi phục lại độ phân giải ban đầu của ảnh gốc.",
      "Càng lúc càng chứa nhiều thông tin chi tiết về các pixel màu sắc của môi trường.",
      "Trở nên ngày càng trừu tượng, mang tính khái niệm cao và chứa ít thông tin thị giác.",
      "Giữ nguyên trạng thái và cấu trúc như lớp đầu tiên mà không có sự thay đổi nào."
    ],
    correctAnswer: 2,
    explanation: "Các biểu diễn ở những lớp sâu hơn mang ít thông tin hình ảnh thực tế (pixel) hơn và chứa nhiều thông tin về khái niệm phân loại (ví dụ: 'mắt mèo', 'tai mèo') hơn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Sự thưa thớt (sparsity) của các kích hoạt thay đổi như thế nào theo độ sâu của lớp ConvNet?",
    options: [
      "Giảm dần, nghĩa là ngày càng có nhiều bộ lọc kích hoạt mạnh với ảnh đầu vào.",
      "Tăng lên, nghĩa là ngày càng có nhiều bộ lọc trả về giá trị rỗng (không kích hoạt).",
      "Sự thưa thớt luôn bằng 0 ở mọi tầng vì dữ liệu luôn được mạng lưới bảo toàn.",
      "Biến đổi ngẫu nhiên hoàn toàn không có tính quy luật nào khi mạng sâu hơn."
    ],
    correctAnswer: 1,
    explanation: "Ở các lớp đầu tiên, hầu hết mọi bộ lọc đều kích hoạt. Ở các lớp sâu, số lượng bộ lọc trống (không phát hiện ra đặc điểm cụ thể đó trong ảnh đầu vào) ngày càng nhiều.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Khái niệm 'Đường ống chưng cất thông tin' (Information distillation pipeline) trong Deep Learning có nghĩa là gì?",
    options: [
      "Quá trình biến đổi mạng thần kinh sâu thành mạng cạn để dễ tính toán hơn.",
      "Chỉ mã hóa các đặc trưng hình ảnh dưới dạng phân phối xác suất ngẫu nhiên.",
      "Dữ liệu thô bị biến đổi liên tục để lọc bỏ thông tin thừa và khuếch đại thông tin hữu ích.",
      "Phương pháp trộn nhiều bức ảnh lại với nhau để tạo ra bộ dữ liệu học khổng lồ."
    ],
    correctAnswer: 2,
    explanation: "Giống như con người nhìn một chiếc xe đạp và chỉ nhớ khái niệm trừu tượng, mạng nơ-ron lọc bỏ chi tiết không liên quan (ví dụ: màu nền) và khuếch đại thông tin hữu ích cho việc phân loại.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Kỹ thuật 'Trực quan hóa bộ lọc ConvNet' (Visualizing ConvNet filters) được thực hiện dựa trên quy trình toán học nào?",
    options: [
      "Sử dụng thuật toán KNN để tìm kiếm bức ảnh giống nhất trong tập huấn luyện.",
      "Gradient Ascent trên không gian ảnh để tối đa hóa sự phản hồi của một bộ lọc.",
      "Sử dụng phép cộng ma trận để gộp tất cả các trọng số của mô hình vào ảnh gốc.",
      "Tính toán đạo hàm bậc hai để triệt tiêu các đặc điểm của ảnh đầu vào ngẫu nhiên."
    ],
    correctAnswer: 1,
    explanation: "Để xem một bộ lọc (filter) nhạy cảm với hình dạng gì, ta khởi tạo một ảnh ngẫu nhiên nhiễu, sau đó dùng Gradient Ascent (tăng dần gradient) trên các pixel của ảnh để làm bộ lọc đó kích hoạt mạnh nhất.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Trong vòng lặp Gradient Ascent để trực quan hóa bộ lọc, tại sao chúng ta cần chuẩn hóa gradient (gradient normalization trick)?",
    options: [
      "Để biến tất cả các giá trị của ảnh đầu vào thành số nguyên giới hạn từ 0-255.",
      "Để chia độ lớn của gradient cho hàm chuẩn L2, giúp bước cập nhật luôn ổn định.",
      "Để buộc tất cả các gradient trở về giá trị 0 nhằm tránh quá khớp overfitting.",
      "Để dịch chuyển hình ảnh sang hệ màu xám giúp tăng tốc độ tính toán đồ họa."
    ],
    correctAnswer: 1,
    explanation: "Chia gradient tensor cho độ lớn L2 norm của nó giúp các bước cập nhật ảnh có biên độ nhất quán, tránh việc cập nhật quá nhỏ hoặc quá lớn làm sai lệch quá trình lặp.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Trong đoạn mã trực quan hóa bộ lọc, sự khác biệt lớn nhất giữa `model.predict(x)` và `model(x)` là gì?",
    options: [
      "`predict()` không hỗ trợ backpropagation, trong khi `model()` cho phép truy xuất gradient.",
      "`predict()` tính toán nhanh hơn gấp trăm lần nhưng kết quả luôn có một sai số nhỏ.",
      "`model()` tự động xử lý hàng loạt lô dữ liệu lớn, còn `predict()` bị tràn RAM.",
      "Hoàn toàn không có sự khác biệt, cả hai hàm thực hiện chính xác một chức năng."
    ],
    correctAnswer: 0,
    explanation: "Khi cần tính gradient (đạo hàm) cho quá trình gradient ascent, ta phải dùng lệnh gọi `model(x)` (trong GradientTape của TF). Lệnh `predict()` không thể dùng để theo dõi tính vi phân.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Kết quả hiển thị sau khi thực hiện quá trình Gradient Ascent trên một bộ lọc cụ thể (filter visualization) là gì?",
    options: [
      "Một ma trận nhị phân chỉ bao gồm các con số 0 và 1 biểu diễn trọng số kết nối mạng.",
      "Một biểu đồ đường xu hướng cho thấy sự suy giảm của hàm loss trong khi huấn luyện.",
      "Một mô hình hình ảnh hoa văn (pattern) đại diện cho những gì bộ lọc phản ứng mạnh nhất.",
      "Bức ảnh gốc của bộ dữ liệu nhưng đã bị áp dụng kỹ thuật làm mờ Gauss (Gaussian blur)."
    ],
    correctAnswer: 2,
    explanation: "Kết quả trả về là một hình ảnh tổng hợp (đã được khử nhiễu) hiển thị cấu trúc hoặc hoa văn (ví dụ: sọc ngang, chấm bi, lông chim) mà bộ lọc đó được thiết kế để tìm kiếm.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Kỹ thuật Grad-CAM (Class Activation Map) chủ yếu trả lời câu hỏi nào về mô hình học sâu?",
    options: [
      "Làm cách nào để nén mô hình xuống mức nhỏ nhất mà không làm giảm độ chính xác?",
      "Tại sao mạng phân loại ảnh này thành con mèo và bộ phận nào khiến nó nghĩ vậy?",
      "Có bao nhiêu bộ lọc đã không được kích hoạt trong toàn bộ quá trình đưa ra dự đoán?",
      "Chính xác mất bao nhiêu mili-giây để GPU có thể kết xuất xong một bức ảnh màu?"
    ],
    correctAnswer: 1,
    explanation: "Grad-CAM trả về một bản đồ nhiệt, đánh dấu vị trí các khu vực trong ảnh đã đóng vai trò quan trọng nhất trong việc khiến mô hình dự đoán ra nhãn cụ thể (ví dụ: phân biệt voi Ấn Độ và voi Châu Phi qua cặp tai).",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Bản chất toán học của Grad-CAM là sự kết hợp của hai yếu tố chính nào?",
    options: [
      "Mức kích hoạt không gian của các kênh (feature maps) VÀ độ quan trọng của kênh đó đối với nhãn.",
      "Hàm mất mát (Loss function) tổng quát VÀ siêu tham số tỷ lệ học tập (Learning rate) cục bộ.",
      "Biểu đồ tần suất Histogram của ảnh đầu vào VÀ kích thước của các ma trận MaxPooling.",
      "Trọng số của lớp Flatten kết nối đầy đủ VÀ giá trị sai số chuẩn trung bình của mô hình."
    ],
    correctAnswer: 0,
    explanation: "Grad-CAM sử dụng đầu ra của lớp Convolution cuối cùng, tính gradient của class đối với từng kênh để tìm 'độ quan trọng' của kênh, rồi nhân độ quan trọng đó với bản đồ kích hoạt không gian.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Trong quy trình tính Grad-CAM, để xác định 'độ quan trọng' của từng kênh (channel) trong lớp tích chập cuối, người ta sử dụng thao tác nào trên gradient?",
    options: [
      "Loại bỏ tất cả gradient âm và chỉ giữ lại các giá trị gradient dương cường độ cao.",
      "Lấy trung bình toàn cầu (Global average pooling) các giá trị gradient trên toàn bộ kênh không gian.",
      "Nhân từng pixel gradient với tham số bias tương ứng của lớp Dense cuối cùng.",
      "Bình phương tất cả các giá trị gradient và lấy căn bậc hai để tránh hiện tượng vỡ."
    ],
    correctAnswer: 1,
    explanation: "Mỗi kênh trong tensor gradient sẽ được gộp trung bình (pooled_grads) theo chiều rộng và cao (spatial dimensions) để trả ra một vector 1D duy nhất chứa mức độ quan trọng của từng kênh.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Để biến bản đồ nhiệt (heatmap) từ chuỗi giá trị số thực thành một lớp phủ hiển thị được trên ảnh gốc, quy trình thường sử dụng kỹ thuật nào?",
    options: [
      "Chuyển mảng số thực về số nguyên 0-255 và ánh xạ thông qua thang màu (như 'jet' colormap).",
      "Chỉ lấy các số dương lớn hơn 1 và gán màu đỏ rực cho tất cả các vùng đó một cách đồng đều.",
      "Xóa bỏ hoàn toàn ảnh gốc và chỉ hiển thị bản đồ nhiệt dưới dạng bản đồ địa hình đơn sắc.",
      "Nhân giá trị bản đồ nhiệt trực tiếp với giá trị hàm kích hoạt ReLU của ảnh đầu vào chuẩn."
    ],
    correctAnswer: 0,
    explanation: "Bản đồ nhiệt được chuẩn hóa về mức 0-255, ánh xạ qua thang màu (ví dụ `cm.get_cmap('jet')`) để tạo ảnh RGB, và cuối cùng nhân với hệ số độ trong suốt (opacity) để chồng lên ảnh gốc.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Mô hình KerasHub `ImageClassifier` giải quyết vấn đề gì giúp việc thiết lập Grad-CAM dễ dàng hơn?",
    options: [
      "Nó loại bỏ mọi quá trình tính toán đạo hàm giúp chạy Grad-CAM trên CPU nhanh chóng.",
      "Nó tự động mã hóa mô hình hộp đen thành mã nguồn C++ để người dùng tinh chỉnh.",
      "Nó gộp sẵn mạng trích xuất đặc trưng, đầu phân loại và bộ tiền xử lý ảnh vào một đối tượng.",
      "Nó thay đổi hoàn toàn kiến trúc mạng ResNet thành Xception chỉ trong một dòng lệnh duy nhất."
    ],
    correctAnswer: 2,
    explanation: "`ImageClassifier` trong KerasHub tích hợp sẵn tiền xử lý ảnh (preprocessing) và phần đầu phân loại (classification head) với nhãn 1000 class ImageNet, giúp luồng dự đoán mượt mà hơn.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Khi quan sát bản đồ Grad-CAM của một chú voi con, nếu mô hình nhận diện chính xác 'Voi Châu Phi', vùng nào của ảnh thường sẽ phát sáng đỏ (được chú ý nhiều nhất)?",
    options: [
      "Toàn bộ vùng bầu trời xanh và đồng cỏ ở phần hậu cảnh bao quanh con voi.",
      "Khu vực chiếc tai lớn đặc trưng giúp mô hình phân biệt với loài voi Ấn Độ.",
      "Những pixel bị nhiễu do máy ảnh tạo ra ở sát các góc viền bức ảnh chụp.",
      "Bốn chân của con voi vì đó là phần thấp nhất và dễ nhận diện nhất của ảnh."
    ],
    correctAnswer: 1,
    explanation: "Voi Châu Phi nổi bật với cặp tai lớn. Lưới nhiệt Grad-CAM thường làm nổi bật phần tai (và có thể là ngà voi), minh chứng cho việc mô hình đã học được 'khái niệm' sinh học chính xác.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Khi thực hiện phương pháp Gradient Ascent để trực quan hóa bộ lọc, bạn nên lặp quy trình (iteration) khoảng bao nhiêu lần là hợp lý theo hướng dẫn?",
    options: [
      "Chỉ cần 1 lần lặp duy nhất là sẽ ra ngay kết quả hoàn chỉnh.",
      "Thường lặp khoảng 30 đến 40 bước (iterations) để hoa văn hội tụ đủ rõ nét.",
      "Ít nhất 10.000 bước để đảm bảo mọi pixel đạt giá trị cực đại không đổi.",
      "Lặp vô hạn cho đến khi GPU cảnh báo tràn bộ nhớ RAM do thuật toán."
    ],
    correctAnswer: 1,
    explanation: "Theo sách (Listing 10.14), số bước lặp thông thường được đặt ở mức `iterations = 30` hoặc 40, đủ để ảnh ngẫu nhiên định hình rõ rệt thành một pattern tối đa hóa activation.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Nhận định nào về quá trình làm mịn hình ảnh trong Gradient Ascent (`deprocess_image`) là chính xác?",
    options: [
      "Nó loại bỏ hàm kích hoạt và thay bằng Batch Normalization ở đầu ra cuối.",
      "Nó chuyển hình ảnh thành đồ thị Vector 3D thay vì lưu dưới định dạng pixel.",
      "Nó cắt bỏ phần biên ảnh (center crop) để tránh các hiệu ứng nhiễu ranh giới (border artifacts).",
      "Nó cộng dồn dữ liệu của tất cả các kênh lại thành một bản đồ đặc trưng 1D."
    ],
    correctAnswer: 2,
    explanation: "Hàm `deprocess_image` (chuẩn hóa về dải 0-255) có bao gồm việc cắt biên `image[25:-25, 25:-25, :]` để loại bỏ các vùng biên thường hay bị nhiễu sóng (border artifacts) trong quá trình backprop.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Sự phân rã đặc trưng của ConvNet có một sự tương đồng lớn với chức năng nào của con người?",
    options: [
      "Hệ thống tự sửa chữa các tế bào máu khi cơ thể mắc bệnh truyền nhiễm.",
      "Hệ thống ghi nhớ hình ảnh chi tiết tuyệt đối của con người trong vô thức.",
      "Khả năng biến các thông tin thị giác thô thành các khái niệm trừu tượng (như nhớ xe đạp).",
      "Sự phân biệt âm lượng tần số cao trong một không gian cực kỳ ồn ào."
    ],
    correctAnswer: 2,
    explanation: "Con người có xu hướng trừu tượng hóa: nhớ rằng có một chiếc 'xe đạp' thay vì nhớ màu sơn chi tiết hay thiết kế khung xe. ConvNet cũng bỏ qua chi tiết pixel và giữ lại khái niệm (concept) qua các lớp.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Tại sao lệnh `tape.watch(image)` là bắt buộc trong mã TensorFlow Grad-CAM?",
    options: [
      "Bởi vì tensor hình ảnh không phải là một `tf.Variable` nên không được theo dõi tự động.",
      "Bởi vì nó ra lệnh cho GPU tự động tải hình ảnh đó lên màn hình người dùng hiển thị.",
      "Bởi vì hình ảnh này được truyền vào từ internet và cần quét kiểm duyệt mã độc rác.",
      "Bởi vì hàm mất mát quá lớn nên cần chặn lại trước khi gây ra lỗi tràn bộ đệm hệ thống."
    ],
    correctAnswer: 0,
    explanation: "Mặc định, `tf.GradientTape()` chỉ tự động theo dõi gradient của các biến có khả năng đào tạo (`tf.Variable` như trọng số lớp mạng). Vì `image` là một tensor hằng số (đầu vào), ta phải yêu cầu tape theo dõi nó thủ công.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Các kỹ thuật diễn giải ConvNet chủ yếu nhắm vào việc cung cấp giá trị gì cho lập trình viên?",
    options: [
      "Khả năng nén kích thước mô hình học sâu xuống mức tính bằng Kilobyte.",
      "Tăng độ chính xác phân loại của mạng trên tập dữ liệu đánh giá lên mức tuyệt đối 100%.",
      "Công cụ để gỡ lỗi (debug) quá trình ra quyết định và bản địa hóa đối tượng (localization).",
      "Biến đổi mọi mạng CNN trở thành một dạng Vision Transformer hiện đại nhất."
    ],
    correctAnswer: 2,
    explanation: "Interpretability giúp developer hiểu được mô hình bị lỗi do nguyên nhân gì (ví dụ: mô hình chó/sói dự đoán sai do học nhầm bối cảnh tuyết), và khả năng khoanh vùng vị trí đối tượng trên ảnh.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Một cách liên hệ, các lớp ConvNet sâu trong mạng phân giải đặc trưng như cách cấu trúc nào phân giải tín hiệu dao động?",
    options: [
      "Thuật toán cây quyết định Random Forest chia tách các mẫu dữ liệu rời rạc.",
      "Thuật toán hồi quy tuyến tính tìm kiếm một đường thẳng gần nhất với dữ liệu.",
      "Phép biến đổi Fourier (Fourier transform) phân rã tín hiệu thành tổ hợp các hàm cosine.",
      "Kiến trúc Transformer xử lý thông tin tự chú ý qua các chuỗi văn bản liên tục."
    ],
    correctAnswer: 2,
    explanation: "ConvNets học một ngân hàng các bộ lọc sao cho mọi đầu vào có thể biểu diễn như sự kết hợp của chúng, rất giống với cách biến đổi Fourier phân rã sóng phức tạp thành các hàm lượng giác cơ bản.",
    difficulty: "Trung bình"
  },
  {
    type: "fill",
    question: "Kỹ thuật {1} (Gradient ascent) được sử dụng để trực quan hóa những mẫu hoa văn trực quan mà bộ lọc ConvNet phản ứng mạnh nhất.",
    blanks: [
      { id: 1, text: "tăng dần gradient", answer: "tăng dần gradient|gradient ascent" }
    ],
    explanation: "Gradient Ascent là kỹ thuật thay đổi giá trị điểm ảnh của bức ảnh nhiễu theo chiều dương của gradient để tối đa hóa sự kích hoạt của một bộ lọc.",
    difficulty: "Trung bình"
  },
  {
    type: "fill",
    question: "Bản đồ nhiệt kích hoạt lớp {1} hiển thị mức độ quan trọng của từng vùng không gian trên ảnh đầu vào đối với quyết định dự đoán nhãn cụ thể.",
    blanks: [
      { id: 1, text: "grad-cam", answer: "grad-cam|grad cam|cam" }
    ],
    explanation: "Grad-CAM (Gradient-weighted Class Activation Mapping) là kỹ thuật quan trọng nhất để tạo ra các heatmap lý giải nguyên nhân dự đoán của mạng CNN.",
    difficulty: "Dễ"
  },
  {
    type: "sorting",
    question: "Sắp xếp thứ tự của quy trình trích xuất bản đồ nhiệt Grad-CAM:",
    steps: [
      "Truyền ảnh đầu vào qua mô hình và tính giá trị kích hoạt của lớp Conv cuối cùng.",
      "Tính đạo hàm (gradient) của lớp dự đoán mục tiêu dựa trên bản đồ kích hoạt trên.",
      "Tính điểm số quan trọng trung bình cho mỗi kênh (Global Average Pooling gradient).",
      "Nhân từng kênh của bản đồ kích hoạt không gian với điểm số quan trọng tương ứng."
    ],
    explanation: "Quy trình chuẩn Grad-CAM: Forward pass tới lớp Conv cuối -> Backward tính Gradient lớp cuối -> Tính trọng số mỗi kênh -> Áp dụng trọng số lên bản đồ không gian để ra Heatmap.",
    difficulty: "Khó"
  },
  {
    type: "matching",
    question: "Ghép nối các API tính gradient với nền tảng học sâu tương ứng khi cài đặt thuật toán Gradient Ascent:",
    pairs: [
      { left: "tape.gradient() trong khối GradientTape", right: "TensorFlow" },
      { left: "hàm .backward() và tensor.grad", right: "PyTorch" },
      { left: "jax.grad()", right: "JAX" },
      { left: "model.predict(x)", right: "Không tính được gradient ở bất kỳ thư viện nào" }
    ],
    explanation: "Đây là điểm cốt lõi để lập trình viên sử dụng các nền tảng Framework trong các vấn đề tùy chỉnh tính toán gradient thấp.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Ghép nối đặc tính của các bộ lọc học được với vị trí tương ứng của chúng trong mô hình ConvNet:",
    pairs: [
      { left: "Bộ lọc học các cạnh, màu sắc đơn sắc, chấm thẳng", right: "Các lớp đầu tiên (Shallow layers)" },
      { left: "Bộ lọc học sự kết hợp của các cạnh tạo thành kết cấu (textures)", right: "Các lớp tầm trung (Middle layers)" },
      { left: "Bộ lọc học các khái niệm đối tượng vật lý như lông chim, tai, lá", right: "Các lớp cuối cùng (Deep layers)" },
      { left: "Hầu hết các bộ lọc trống trơn (không phản ứng với ảnh đầu vào)", right: "Độ thưa thớt (Sparsity) cao khi mạng càng sâu" }
    ],
    explanation: "Nguyên lý biểu diễn phân cấp là một tính chất phổ quát của mọi mạng thần kinh sâu: từ chi tiết cấp thấp đến ngữ nghĩa cấp cao.",
    difficulty: "Dễ"
  }
];

export default questions;
