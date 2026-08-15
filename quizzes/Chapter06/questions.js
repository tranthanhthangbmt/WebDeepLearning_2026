const quizData = [
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 1: Ba giai đoạn chính trong quy trình làm việc phổ biến của học máy là gì?",
    options: [
      "Thu thập dữ liệu, viết mã nguồn Python, và thiết kế giao diện ứng dụng.",
      "Xác định nhiệm vụ, Phát triển mô hình, và Triển khai mô hình.",
      "Chuẩn hóa dữ liệu, Đào tạo mạng nơ-ron sâu, và Tối ưu hóa siêu tham số.",
      "Xây dựng kiến trúc mô hình, Chạy thử nghiệm, và Khắc phục lỗi."
    ],
    correctAnswer: 1,
    explanation: "Quy trình chuẩn bao gồm: (1) Xác định nhiệm vụ (hiểu vấn đề, thu thập dữ liệu), (2) Phát triển mô hình (huấn luyện, đánh giá) và (3) Triển khai mô hình (đưa vào thực tế và bảo trì)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 2: Khi bắt đầu một dự án học máy thực tế, yếu tố nào thường là điểm giới hạn lớn nhất và đòi hỏi nhiều nỗ lực nhất?",
    options: [
      "Việc lựa chọn framework học sâu phù hợp như TensorFlow hay PyTorch.",
      "Việc tìm kiếm một card đồ họa (GPU) đủ mạnh để huấn luyện mô hình.",
      "Sự sẵn có của dữ liệu huấn luyện và quá trình thu thập, gán nhãn dữ liệu.",
      "Quá trình triển khai mô hình lên các máy chủ web thông qua REST API."
    ],
    correctAnswer: 2,
    explanation: "Dữ liệu thường là nút thắt cổ chai lớn nhất. Việc thu thập dữ liệu phù hợp và gán nhãn thủ công tốn rất nhiều thời gian, công sức và chi phí so với việc thiết kế thuật toán."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 3: Dự án 'Dự đoán tỷ lệ nhấp vào quảng cáo hiển thị' (Click-through rate) thuộc loại bài toán học máy nào?",
    options: [
      "Phân loại nhị phân (Binary classification)",
      "Hồi quy vô hướng (Scalar regression)",
      "Phân loại đa lớp (Multiclass classification)",
      "Hồi quy vectơ (Vector regression)"
    ],
    correctAnswer: 1,
    explanation: "Dự đoán tỷ lệ nhấp chuột là bài toán dự đoán một giá trị số liên tục (ví dụ: 0.15 hoặc 15%), do đó nó là bài toán Hồi quy vô hướng (Scalar regression)."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 4: Khi xây dựng một hệ thống AI để 'đánh giá độ tin cậy của ai đó từ hình ảnh khuôn mặt', rủi ro lớn nhất về mặt đạo đức là gì?",
    options: [
      "Mô hình sẽ mất quá nhiều thời gian để hội tụ do đặc điểm khuôn mặt rất phức tạp.",
      "Hệ thống sẽ vô tình học và mã hóa các định kiến, thành kiến của người gán nhãn.",
      "Ứng dụng sẽ vi phạm các chính sách chia sẻ tài nguyên mã nguồn mở hiện hành.",
      "Độ chính xác của mạng nơ-ron sẽ giảm mạnh khi áp dụng trên các thiết bị di động."
    ],
    correctAnswer: 1,
    explanation: "Công nghệ không trung tính. Các mô hình như vậy sẽ học thuộc các định kiến của con người (người gán nhãn) và bọc chúng dưới vỏ bọc 'AI khách quan', gây ra các quyết định thiên lệch và bất công trong đời thực."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 5: Trong học máy, khái niệm 'Data leakage' (Rò rỉ mục tiêu/Rò rỉ dữ liệu) là gì?",
    options: [
      "Sự cố hệ thống khiến cho bộ dữ liệu bị sao chép trái phép ra ngoài internet.",
      "Việc dữ liệu huấn luyện chứa thông tin về mục tiêu mà trong thực tế sẽ không có.",
      "Trường hợp dữ liệu bị thiếu hụt một lượng lớn các tính năng quan trọng.",
      "Việc các nhà phát triển vô tình xóa nhầm tập dữ liệu kiểm tra cuối cùng."
    ],
    correctAnswer: 1,
    explanation: "Rò rỉ mục tiêu xảy ra khi tập huấn luyện chứa các đặc trưng trực tiếp tiết lộ kết quả mục tiêu (ví dụ: dùng đặc trưng 'đã từng điều trị ung thư' để dự đoán bệnh ung thư trong tương lai)."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 6: Hiện tượng dữ liệu sản xuất thay đổi đặc điểm theo thời gian, khiến độ chính xác của mô hình giảm dần được gọi là sự trôi dạt ________.",
    blanks: ["khái niệm", "concept", "concept drift", "khái niệm (concept drift)"],
    explanation: "Concept drift (Trôi dạt khái niệm) là hiện tượng phổ biến (như ngôn ngữ thay đổi, xu hướng tiêu dùng thay đổi) khiến một mô hình từng hoạt động rất tốt trong quá khứ bị lỗi thời."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 7: Sai lệch lấy mẫu (Sampling bias) xảy ra khi nào?",
    options: [
      "Khi quá trình thu thập dữ liệu bị đình trệ do thiếu nhân lực và kinh phí.",
      "Khi mô hình học sâu có kiến trúc không phù hợp với định dạng dữ liệu đầu vào.",
      "Khi tập dữ liệu huấn luyện được xáo trộn một cách hoàn toàn ngẫu nhiên.",
      "Khi quá trình thu thập không tạo ra một mẫu đại diện ngẫu nhiên cho thực tế."
    ],
    correctAnswer: 3,
    explanation: "Sai lệch lấy mẫu (như cuộc bầu cử Mỹ 1948) xảy ra khi phương pháp thu thập dữ liệu tự động loại trừ hoặc thiên vị một nhóm đối tượng, khiến tập dữ liệu không phản ánh đúng quần thể thực tế."
  },
  {
    type: "matching",
    difficulty: "Trung bình",
    question: "Câu 8: Ghép nối các loại bài toán với Hàm Loss (Loss function) phù hợp nhất:",
    pairs: [
      { left: "Phân loại nhị phân (Binary classification)", right: "Binary crossentropy" },
      { left: "Phân loại đa lớp (Multiclass, single-label)", right: "Categorical crossentropy" },
      { left: "Hồi quy (Regression)", right: "Mean squared error (MSE)" }
    ],
    explanation: "Đây là các hàm mất mát tiêu chuẩn: Binary crossentropy cho 2 lớp, Categorical crossentropy cho nhiều lớp độc quyền, và MSE cho việc dự đoán giá trị liên tục."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 9: Bước đầu tiên cần làm trong giai đoạn 'Phát triển mô hình' (sau khi có dữ liệu) là gì?",
    options: [
      "Thử nghiệm kiến trúc mạng nơ-ron phức tạp nhất có thể tìm thấy.",
      "Chuẩn bị và tiền xử lý dữ liệu (Vector hóa, chuẩn hóa, xử lý thiếu).",
      "Viết các bài kiểm thử đơn vị (Unit tests) cho ứng dụng di động.",
      "Cài đặt thuật toán Early Stopping để tối ưu hóa quá trình học."
    ],
    correctAnswer: 1,
    explanation: "Trước khi đưa dữ liệu vào mạng nơ-ron, bắt buộc phải tiền xử lý nó: chuyển đổi thành vector số (tensor), đưa các giá trị về quy mô nhỏ (chuẩn hóa), và xử lý các điểm dữ liệu bị khuyết thiếu."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 10: Theo quy tắc chuẩn hóa dữ liệu chặt chẽ, ta nên chuẩn hóa các đặc trưng (features) để chúng có giá trị trung bình (mean) và độ lệch chuẩn (standard deviation) bằng bao nhiêu?",
    options: [
      "Giá trị trung bình bằng 1, độ lệch chuẩn bằng 0.",
      "Giá trị trung bình bằng 0, độ lệch chuẩn bằng 1.",
      "Giá trị trung bình bằng 0.5, độ lệch chuẩn bằng 10.",
      "Giá trị trung bình bằng 0, độ lệch chuẩn không thay đổi."
    ],
    correctAnswer: 1,
    explanation: "Quy tắc chuẩn hóa tiêu chuẩn (Standardization) là trừ đi mean (để mean = 0) và chia cho độ lệch chuẩn (để std = 1). Điều này giúp các nơ-ron học dễ dàng và nhanh chóng hơn."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 11: Khi gặp đặc trưng (feature) dạng số bị thiếu giá trị trong dữ liệu, cách xử lý nào sau đây là TỐT NHẤT cho mạng nơ-ron?",
    options: [
      "Thay thế toàn bộ bằng số 0 để mạng nơ-ron tự động bỏ qua chúng.",
      "Xóa bỏ hoàn toàn cột đặc trưng đó khỏi tập dữ liệu huấn luyện.",
      "Thay thế bằng giá trị trung bình hoặc trung vị của đặc trưng đó.",
      "Tạo thêm một mô hình học máy khác để phân loại lại dữ liệu thiếu."
    ],
    correctAnswer: 2,
    explanation: "Điền số 0 một cách tùy tiện có thể tạo ra sự gián đoạn trong không gian đặc trưng. Cách tốt nhất và an toàn nhất thường là thay thế bằng giá trị trung bình (mean) hoặc trung vị (median) của dữ liệu hiện có."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 12: Đâu là mục tiêu quan trọng đầu tiên khi bạn bắt đầu viết mã cho kiến trúc mô hình của mình?",
    options: [
      "Đạt được sức mạnh thống kê (đánh bại một đường cơ sở đơn giản).",
      "Đạt được trạng thái Overfitting ngay trong lần chạy thử nghiệm đầu.",
      "Áp dụng thành công tất cả các kỹ thuật chuẩn hóa (Regularization).",
      "Làm cho mô hình chạy nhanh nhất có thể trên trình duyệt web."
    ],
    correctAnswer: 0,
    explanation: "Trước khi nghĩ đến những mô hình phức tạp, bạn phải chứng minh được rằng bài toán có thể giải quyết được bằng học máy. Mục tiêu đầu tiên là tạo một mô hình nhỏ có khả năng vượt qua một 'common-sense baseline' (đường cơ sở ngây thơ)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 13: Sau khi đã có một mô hình vượt qua đường cơ sở (baseline), bước TIẾP THEO trong quy trình phát triển mô hình là gì?",
    options: [
      "Triển khai ngay mô hình đó lên máy chủ web qua REST API.",
      "Tiến hành cắt tỉa trọng số (Weight pruning) để thu gọn mạng.",
      "Mở rộng mô hình (Scale up) để đạt được trạng thái trang bị quá mức.",
      "Triển khai thuật toán kiểm tra A/B (A/B testing) với người dùng."
    ],
    correctAnswer: 2,
    explanation: "Theo quy trình, để biết mô hình cần độ phức tạp bao nhiêu là đủ, bạn phải xây dựng một mô hình lớn (thêm layer, thêm nơ-ron) cho đến khi nó vượt qua ranh giới khả năng khái quát và bắt đầu Overfit."
  },
  {
    type: "sorting",
    difficulty: "Trung bình",
    question: "Câu 14: Sắp xếp các bước chuẩn khi Phát triển một mô hình học sâu:",
    steps: [
      "Chuẩn bị dữ liệu (Tiền xử lý, chuẩn hóa).",
      "Xây dựng mô hình đơn giản để đánh bại đường cơ sở (Baseline).",
      "Mở rộng quy mô mô hình (Scale up) để đạt trạng thái Overfit.",
      "Áp dụng Chuẩn hóa (Regularization) và Tinh chỉnh siêu tham số."
    ],
    explanation: "Quy trình phát triển nội bộ: Data Prep -> Đạt sức mạnh thống kê (vượt Baseline) -> Ép mô hình Overfit -> Tinh chỉnh và Regularization để đạt khả năng khái quát tốt nhất."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 15: Tại sao chúng ta không thể sử dụng trực tiếp các số liệu như ROC AUC để làm Hàm mất mát (Loss function)?",
    options: [
      "Vì ROC AUC chỉ có thể được tính toán vào cuối toàn bộ quá trình huấn luyện.",
      "Vì ROC AUC không có khả năng phân biệt giữa các lớp dữ liệu khác nhau.",
      "Vì ROC AUC không khả vi (not differentiable) và không tính được trên mini-batch.",
      "Vì ROC AUC có dải giá trị quá lớn, khiến thuật toán cập nhật bị tràn bộ nhớ."
    ],
    correctAnswer: 2,
    explanation: "Hàm mất mát bắt buộc phải tính được toán đạo hàm (differentiable) để có thể thực hiện thuật toán lan truyền ngược (backpropagation), và phải tính toán ổn định trên một mini-batch nhỏ. ROC AUC không đáp ứng được các yếu tố toán học này."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 16: Nếu mô hình liên tục được tinh chỉnh dựa trên phản hồi từ tập dữ liệu xác thực (Validation set), điều gì sẽ xảy ra?",
    options: [
      "Mô hình sẽ đạt được độ chính xác tuyệt đối 100% trên dữ liệu thực tế.",
      "Sẽ xảy ra hiện tượng Overfitting ngầm vào chính tập dữ liệu xác thực.",
      "Tập dữ liệu xác thực sẽ tự động thay đổi các giá trị nhãn ban đầu.",
      "Quá trình tính toán đạo hàm sẽ trở nên tuyến tính và nhanh chóng hơn."
    ],
    correctAnswer: 1,
    explanation: "Dù mô hình không học trực tiếp từ Validation set, nhưng khi bạn tinh chỉnh siêu tham số nhiều lần dựa trên điểm số của nó, bạn đã làm rò rỉ thông tin. Kết quả là mô hình sẽ bị Overfit ngầm vào Validation set, làm kết quả đánh giá không còn đáng tin cậy."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 17: Khi trình bày mô hình cho các bên liên quan (stakeholders) trước khi ra mắt, hành động nào là CẦN THIẾT?",
    options: [
      "Giấu nhẹm đi mọi trường hợp dự đoán sai để tránh làm khách hàng lo lắng.",
      "Khẳng định mô hình sẽ đạt được tỷ lệ chính xác tuyệt đối như con người.",
      "Thiết lập kỳ vọng thực tế, trình bày rõ các tỷ lệ sai sót và rủi ro.",
      "Chỉ sử dụng mã nguồn Python nguyên bản cho tất cả các bản demo."
    ],
    correctAnswer: 2,
    explanation: "Không có mô hình nào hoàn hảo. Việc quản lý kỳ vọng là tối quan trọng: hãy trình bày rõ các chế độ hỏng hóc (failure modes), tỷ lệ dương tính giả (false positive) và kết nối chúng với bài toán kinh doanh."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 18: Khi cần triển khai một mô hình để kiểm tra thư rác trên ứng dụng nhắn tin được mã hóa đầu cuối (end-to-end encrypted), giải pháp triển khai nào là BẮT BUỘC?",
    options: [
      "Triển khai mô hình dưới dạng một REST API trên máy chủ đám mây.",
      "Triển khai mô hình ngay trên thiết bị của người dùng (smartphone).",
      "Sử dụng kỹ thuật phân bổ tải tự động qua dịch vụ bên thứ ba.",
      "Yêu cầu ứng dụng gửi dữ liệu văn bản thô trực tiếp về cơ sở dữ liệu."
    ],
    correctAnswer: 1,
    explanation: "Vì tin nhắn được mã hóa đầu cuối, máy chủ hoàn toàn không thể đọc được nội dung tin nhắn. Do đó, mô hình AI bắt buộc phải được đẩy xuống thiết bị cục bộ (điện thoại người dùng) để thực hiện dự đoán ngoại tuyến."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 19: Công nghệ nào của hệ sinh thái TensorFlow cho phép mô hình học sâu chạy trực tiếp trên trình duyệt web thông qua JavaScript?",
    options: [
      "TensorFlow Lite",
      "TensorFlow Serving",
      "TensorFlow.js",
      "TensorFlow Extended"
    ],
    correctAnswer: 2,
    explanation: "TensorFlow.js là thư viện hỗ trợ xuất và chạy mô hình học sâu trực tiếp bằng JavaScript trên các trình duyệt web hoặc ứng dụng desktop dùng Electron."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 20: TensorFlow Lite (TF Lite) được thiết kế chủ yếu cho môi trường triển khai nào?",
    options: [
      "Các cụm siêu máy tính dùng để huấn luyện mô hình ngôn ngữ lớn.",
      "Máy chủ web lưu trữ và phân phối mô hình học sâu (REST API).",
      "Các thiết bị di động (Android/iOS) và hệ thống nhúng (Raspberry Pi).",
      "Môi trường phân tích dữ liệu trực tiếp trong trình duyệt web."
    ],
    correctAnswer: 2,
    explanation: "TF Lite là giải pháp tối ưu hóa cực kỳ nhẹ, dùng cho việc chạy suy luận (inference) học sâu trên thiết bị di động, vi điều khiển, và hệ thống nhúng với nguồn tài nguyên phần cứng hạn chế."
  },
  {
    type: "fill_blank",
    difficulty: "Khó",
    question: "Câu 21: Để tối ưu hóa mô hình khi triển khai lên thiết bị di động, kỹ thuật chuyển đổi trọng số từ định dạng float32 sang số nguyên int8 được gọi là Lượng tử hóa trọng số (Weight ________).",
    blanks: ["quantization", "quantize", "lượng tử hóa", "lượng tử"],
    explanation: "Weight quantization (Lượng tử hóa trọng số) là kỹ thuật nén mô hình phổ biến. Nó giảm độ chính xác của trọng số từ số thực 32-bit xuống số nguyên 8-bit, giúp thu nhỏ mô hình gấp 4 lần mà ít làm giảm độ chính xác."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 22: Kỹ thuật 'Cắt tỉa trọng số' (Weight pruning) trong giai đoạn tối ưu hóa mô hình có mục đích gì?",
    options: [
      "Loại bỏ những nơ-ron bị hỏng phần cứng khi triển khai mô hình lên các thiết bị nhúng.",
      "Tự động loại bỏ toàn bộ tập dữ liệu kiểm tra để giải phóng không gian bộ nhớ lưu trữ.",
      "Giữ lại các trọng số quan trọng nhất và loại bỏ những trọng số ít đóng góp cho dự đoán.",
      "Mã hóa các trọng số thành chuẩn âm thanh để mô hình có thể phát ra cảnh báo."
    ],
    correctAnswer: 2,
    explanation: "Trong một mạng nơ-ron lớn, nhiều kết nối có giá trị xấp xỉ 0 và không ảnh hưởng nhiều đến dự đoán. Việc 'cắt tỉa' (ép về 0 và xóa bỏ) giúp thu gọn mô hình đáng kể, làm tăng tốc độ suy luận."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 23: Cách tốt nhất để đánh giá tác động thực sự của một hệ thống AI mới (như Hệ thống đề xuất) lên các chỉ số kinh doanh là gì?",
    options: [
      "Chỉ dựa vào sự cải thiện của chỉ số ROC AUC trên tập dữ liệu kiểm tra cục bộ.",
      "Hỏi trực tiếp những kỹ sư thiết kế xem hệ thống mới hoạt động tốt đến mức nào.",
      "Thực hiện kiểm tra A/B ngẫu nhiên (A/B testing) giữa hệ thống mới và quy trình cũ.",
      "Yêu cầu tất cả khách hàng phải để lại đánh giá bằng văn bản dài sau mỗi lần sử dụng."
    ],
    correctAnswer: 2,
    explanation: "Thử nghiệm A/B là tiêu chuẩn vàng. Chỉ bằng cách phục vụ hệ thống mới cho nhóm A và hệ thống cũ cho nhóm B trong điều kiện thực tế, ta mới có thể đo lường chính xác tác động kinh doanh (tỷ lệ click, thời gian xem, v.v.)."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 24: Khi triển khai mô hình qua REST API, hạn chế lớn nhất của phương pháp này là gì?",
    options: [
      "Khó khăn trong việc cập nhật phiên bản mới của mô hình mạng nơ-ron sâu.",
      "Ứng dụng hoàn toàn phụ thuộc vào việc kết nối internet một cách liên tục.",
      "Đòi hỏi thiết bị di động của người dùng phải có cấu hình cực kỳ mạnh mẽ.",
      "Mô hình không thể sử dụng để xử lý các thuật toán phân loại nhiều lớp."
    ],
    correctAnswer: 1,
    explanation: "Vì suy luận được chạy trên máy chủ đám mây, nếu thiết bị khách hàng bị mất kết nối internet (chế độ máy bay, sóng yếu), toàn bộ tính năng AI của ứng dụng sẽ ngừng hoạt động."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 25: Nếu mô hình có yêu cầu cực kỳ khắt khe về độ trễ thấp (low-latency) và cần xử lý dữ liệu video thời gian thực, bạn nên triển khai nó ở đâu?",
    options: [
      "Trên một nền tảng REST API thông qua Google Cloud Storage.",
      "Chạy mô hình trực tiếp trên trình duyệt bằng TensorFlow.js.",
      "Trực tiếp trên thiết bị (on-device) xử lý hình ảnh tại chỗ.",
      "Lưu trữ trong cơ sở dữ liệu để truy vấn định kỳ mỗi phút."
    ],
    correctAnswer: 2,
    explanation: "Đối với yêu cầu thời gian thực, việc gửi video lên đám mây và đợi kết quả trả về qua REST API sẽ tốn hàng trăm mili-giây (latency cao). Do đó, mô hình cần được nhúng thẳng vào thiết bị máy ảnh hoặc thiết bị tại chỗ (on-device)."
  },
  {
    type: "matching",
    difficulty: "Trung bình",
    question: "Câu 26: Ghép nối phương pháp triển khai với tình huống sử dụng phù hợp:",
    pairs: [
      { left: "REST API", right: "Sử dụng cho hệ thống máy chủ, không yêu cầu độ trễ quá khắt khe." },
      { left: "On-device (TF Lite)", right: "Dữ liệu nhạy cảm cần bảo mật cục bộ, chạy offline." },
      { left: "Browser (TF.js)", right: "Tận dụng GPU của máy tính người dùng trực tiếp trên trình duyệt web." }
    ],
    explanation: "REST API dùng khi cần sức mạnh máy chủ. On-device dùng khi yêu cầu offline, bảo mật, và độ trễ thấp. TF.js giúp ứng dụng web không tốn phí máy chủ suy luận."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 27: Tại sao việc đánh giá mô hình bằng phương pháp 'K-fold cross-validation' lại được ưu tiên thay vì 'Hold-out validation' trong một số trường hợp?",
    options: [
      "Vì nó tiết kiệm được tài nguyên tính toán của card đồ họa.",
      "Vì tập dữ liệu có kích thước quá nhỏ, cần đánh giá ổn định hơn.",
      "Vì nó tự động gán lại nhãn cho các điểm dữ liệu bị sai lệch.",
      "Vì nó là yêu cầu bắt buộc của tất cả các thư viện học sâu."
    ],
    correctAnswer: 1,
    explanation: "Khi lượng dữ liệu quá ít, việc tách hẳn một phần làm tập Validation (Hold-out) sẽ gây ra sự biến động kết quả lớn. K-fold giúp tận dụng toàn bộ dữ liệu để đánh giá mô hình một cách khách quan hơn."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 28: Đạo đức trong học máy đòi hỏi các kỹ sư khi xây dựng hệ thống phải chú ý điều gì?",
    options: [
      "Phải mã hóa các tính năng dự đoán bằng những ngôn ngữ lập trình mã nguồn mở.",
      "Luôn luôn sử dụng tập dữ liệu có nguồn gốc từ nền tảng Kaggle cho an toàn.",
      "Nhận thức rõ những thành kiến trong dữ liệu và không mù quáng số hóa sự phân biệt đối xử.",
      "Đảm bảo mô hình chạy được trên cả thiết bị chạy Android lẫn thiết bị chạy iOS."
    ],
    correctAnswer: 2,
    explanation: "AI học từ dữ liệu của con người. Kỹ sư phải có trách nhiệm đảm bảo hệ thống không học và tự động hóa những định kiến, sự bất công từ quá khứ áp dụng lên thực tại."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 29: Để duy trì mô hình (Maintaining) sau khi đã triển khai lên môi trường sản xuất, công việc thiết yếu là gì?",
    options: [
      "Liên tục thu thập dữ liệu mới, cập nhật nhãn và chuẩn bị đào tạo mô hình thế hệ tiếp theo.",
      "Dừng mọi hoạt động thu thập dữ liệu do mô hình đã hoàn thành xong nhiệm vụ của nó.",
      "Chuyển toàn bộ hệ thống sang sử dụng phương pháp kiểm tra A/B thử nghiệm vĩnh viễn.",
      "Mã hóa cứng mọi quy luật do mạng nơ-ron tạo ra vào các câu lệnh if/else truyền thống."
    ],
    correctAnswer: 0,
    explanation: "Do hiện tượng trôi dạt khái niệm (concept drift), không có mô hình học máy nào tồn tại vĩnh viễn. Đội ngũ AI phải liên tục thu thập dữ liệu mới và huấn luyện các phiên bản cập nhật."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 30: 'Mô hình học máy chỉ có thể nhận ra những gì nó từng thấy'. Nhận định này mang lại hệ quả nào khi ứng dụng mô hình để dự đoán tương lai?",
    options: [
      "Mô hình học sâu tự động phát sinh khả năng lý luận triết học khi đạt đủ tham số.",
      "Dự đoán tương lai dựa trên giả định cốt lõi rằng tương lai sẽ vận hành giống hệt quá khứ.",
      "Các đặc trưng chuẩn hóa luôn đảm bảo mô hình có thể giải quyết được bất kỳ sự kiện mới nào.",
      "Mạng nơ-ron luôn cần phải có thêm thuật toán học tăng cường để hỗ trợ cho việc này."
    ],
    correctAnswer: 1,
    explanation: "Machine learning thực chất là phép nội suy và ghi nhớ mô hình. Khi dùng nó để dự đoán giá chứng khoán hay hành vi người dùng, ta đang đánh cược vào giả định: tương lai không khác gì quá khứ. Khi thế giới thay đổi (Covid-19, khủng hoảng kinh tế), mô hình sẽ thất bại."
  }
];

export default quizData;
