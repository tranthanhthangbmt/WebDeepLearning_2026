const quizData = [
  // Topic 1: Trí tuệ nhân tạo, học máy và học sâu (Mối quan hệ)
  {
    type: "mcq",
    question: "Câu 1: Mối quan hệ bao hàm giữa Trí tuệ nhân tạo (AI), Học máy (ML) và Học sâu (DL) được biểu diễn chính xác nhất như thế nào?",
    options: [
      "AI bao hàm ML, và ML bao hàm DL.",
      "DL bao hàm ML, và ML bao hàm AI.",
      "ML là tập hợp giao nhau giữa AI và DL.",
      "AI, ML và DL là ba lĩnh vực hoàn toàn độc lập."
    ],
    correctAnswer: 0,
    explanation: "Trí tuệ nhân tạo (AI) là lĩnh vực rộng nhất, bao gồm Học máy (ML). Học sâu (DL) là một nhánh chuyên sâu nằm bên trong Học máy.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Câu 2: Đâu là định nghĩa súc tích nhất về Trí tuệ nhân tạo (AI) được nêu trong tài liệu?",
    options: [
      "Nỗ lực tự động hóa các tác vụ trí tuệ thường do con người thực hiện.",
      "Sự phát triển các thuật toán toán học tiên tiến dựa trên mạng lưới thần kinh.",
      "Lĩnh vực thiết kế người máy để thay thế hoàn toàn lao động thủ công.",
      "Phương pháp tạo ra các chương trình có khả năng viết mã lập trình độc lập."
    ],
    correctAnswer: 0,
    explanation: "AI ra đời vào những năm 1950 như một nỗ lực chung nhằm tự động hóa các tác vụ trí tuệ mà bình thường chỉ con người mới thực hiện được.",
    difficulty: "Dễ"
  },
  {
    type: "fill_blank",
    question: "Câu 3: Điền từ thích hợp: Trong những ngày đầu, trí tuệ nhân tạo tin rằng trí thông minh của con người có thể đạt được thông qua việc lập trình viên tạo ra một bộ quy tắc khổng lồ. Cách tiếp cận này được gọi là AI _________.",
    blanks: ["biểu tượng"],
    explanation: "AI biểu tượng (symbolic AI) là mô hình thống trị từ những năm 1950 đến cuối 1980, dựa trên việc lập trình các quy tắc (rules) một cách rõ ràng.",
    difficulty: "Trung bình"
  },

  // Topic 2: Trí tuệ nhân tạo (Symbolic AI)
  {
    type: "mcq",
    question: "Câu 4: AI biểu tượng (Symbolic AI) đã chứng minh được sự phù hợp và hiệu quả cao nhất trong việc giải quyết loại bài toán nào sau đây?",
    options: [
      "Các bài toán logic được xác định rõ ràng, ví dụ như chơi cờ vua.",
      "Phân loại hình ảnh độ phân giải cao trong y tế.",
      "Nhận dạng và dịch thuật ngôn ngữ tự nhiên theo thời gian thực.",
      "Điều khiển xe tự lái trong điều kiện giao thông đông đúc."
    ],
    correctAnswer: 0,
    explanation: "AI biểu tượng hoạt động tốt đối với các vấn đề có quy tắc logic rõ ràng như cờ vua. Tuy nhiên, nó thất bại với các vấn đề phức tạp, mờ nhạt như nhận dạng hình ảnh hay giọng nói.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Câu 5: Tại sao AI biểu tượng cuối cùng lại bị thay thế bởi Học máy trong các tác vụ nhận thức phức tạp?",
    options: [
      "Không thể tìm ra các quy tắc rõ ràng để giải quyết các vấn đề phức tạp như dịch thuật hoặc nhận dạng hình ảnh.",
      "Việc viết mã cho AI biểu tượng đòi hỏi quá nhiều chi phí phần cứng và thời gian tính toán.",
      "Các ngôn ngữ lập trình thời đó không đủ sức mạnh để xử lý các thuật toán biểu tượng.",
      "AI biểu tượng chỉ có thể chạy trên siêu máy tính chuyên dụng, không phổ biến."
    ],
    correctAnswer: 0,
    explanation: "Đối với các bài toán mờ như nhận dạng giọng nói hay phân loại ảnh, con người không thể định nghĩa và mã hóa rõ ràng tất cả các quy tắc, khiến AI biểu tượng trở nên bất lực.",
    difficulty: "Khó"
  },
  {
    type: "matching",
    question: "Câu 6: Ghép nối các khái niệm với đặc điểm tương ứng của chúng:",
    pairs: [
      { left: "Trí tuệ nhân tạo (AI)", right: "Bao gồm cả các phương pháp không cần 'học' (như viết luật)." },
      { left: "AI biểu tượng", right: "Dựa vào việc lập trình viên tự viết ra các quy tắc logic." },
      { left: "Học máy (ML)", right: "Hệ thống tự tìm ra quy tắc dựa trên dữ liệu và kết quả." }
    ],
    explanation: "AI là khái niệm rộng nhất. AI biểu tượng sử dụng các quy tắc được định nghĩa trước (không có học). Học máy thay đổi mô hình bằng cách cho máy tự học quy tắc từ dữ liệu.",
    difficulty: "Trung bình"
  },

  // Topic 3: Học máy
  {
    type: "mcq",
    question: "Câu 7: Sự khác biệt cơ bản nhất về luồng thông tin (inputs/outputs) giữa Lập trình truyền thống và Học máy là gì?",
    options: [
      "Lập trình truyền thống: Dữ liệu + Quy tắc -> Câu trả lời. Học máy: Dữ liệu + Câu trả lời -> Quy tắc.",
      "Lập trình truyền thống: Câu trả lời + Quy tắc -> Dữ liệu. Học máy: Dữ liệu + Câu trả lời -> Quy tắc.",
      "Lập trình truyền thống: Dữ liệu -> Quy tắc + Câu trả lời. Học máy: Quy tắc -> Dữ liệu + Câu trả lời.",
      "Cả hai đều nhận Dữ liệu và Quy tắc để xuất ra Câu trả lời, nhưng Học máy nhanh hơn."
    ],
    correctAnswer: 0,
    explanation: "Trong học máy, chúng ta cung cấp dữ liệu đầu vào cùng với câu trả lời mong đợi, và hệ thống sẽ tự động tìm ra các quy tắc (rules) phù hợp. Điều này ngược lại với lập trình truyền thống.",
    difficulty: "Trung bình"
  },
  {
    type: "fill_blank",
    question: "Câu 8: Điền từ thích hợp: Một hệ thống học máy được đào tạo chứ không được lập trình rõ ràng. Nó đòi hỏi nhiều ví dụ thực tế liên quan đến nhiệm vụ để có thể tìm ra _________.",
    blanks: ["quy tắc", "luật"],
    explanation: "Hệ thống học máy học từ ví dụ (dữ liệu + đáp án) để tự động trích xuất cấu trúc thống kê và tạo ra các quy tắc tự động hóa tác vụ.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Câu 9: Yếu tố nào sau đây bắt đầu xuất hiện vào những năm 1990 giúp Học máy thực sự bùng nổ và bắt đầu tỏa sáng?",
    options: [
      "Sự sẵn có của phần cứng tính toán nhanh hơn và các tập dữ liệu lớn hơn.",
      "Sự ra đời của ngôn ngữ lập trình Python và hệ sinh thái thư viện của nó.",
      "Những đột phá lý thuyết đầu tiên về AI biểu tượng và quy tắc logic.",
      "Việc phát minh ra vi mạch máy tính cơ học đầu tiên."
    ],
    correctAnswer: 0,
    explanation: "Máy học đòi hỏi dữ liệu và sức mạnh tính toán. Sự phát triển mạnh mẽ của phần cứng và khối lượng dữ liệu khổng lồ (internet) vào thập niên 90 đã mở đường cho ML bùng nổ.",
    difficulty: "Trung bình"
  },

  // Topic 4: Học các quy tắc và biểu diễn từ dữ liệu
  {
    type: "mcq",
    question: "Câu 10: Trọng tâm cốt lõi của Học máy là tìm kiếm các \"biểu diễn\" (representations) hữu ích. Một biểu diễn tốt có tác dụng chính là gì?",
    options: [
      "Làm cho dữ liệu có ý nghĩa hơn, giúp nhiệm vụ hiện tại trở nên dễ dàng giải quyết hơn.",
      "Nén dữ liệu đầu vào để giảm thiểu dung lượng lưu trữ trên bộ nhớ hệ thống.",
      "Chuyển đổi dữ liệu văn bản sang dữ liệu âm thanh để tăng cường khả năng tương tác.",
      "Trực quan hóa dữ liệu dưới dạng đồ thị 3D cho các nhà khoa học dữ liệu."
    ],
    correctAnswer: 0,
    explanation: "Một biểu diễn là một cách nhìn khác về dữ liệu. Chẳng hạn như chuyển từ tọa độ Descartes sang hệ tọa độ cực có thể giúp việc phân chia các điểm dữ liệu trở nên cực kỳ đơn giản.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Câu 11: Ghép nối các khái niệm về quá trình Học máy với mô tả của chúng:",
    pairs: [
      { left: "Điểm dữ liệu đầu vào", right: "Hình ảnh kỹ thuật số hoặc tệp âm thanh cần xử lý." },
      { left: "Ví dụ về đầu ra dự kiến", right: "Nhãn 'Mèo' hoặc 'Chó' được gán cho một hình ảnh." },
      { left: "Biểu diễn (Representation)", right: "Cách biến đổi dữ liệu để phân loại dễ dàng hơn." },
      { left: "Không gian giả thuyết", right: "Tập hợp tất cả các phép biến đổi khả thi mà mô hình có thể thử." }
    ],
    explanation: "Quá trình học máy cần Input (dữ liệu), Output (kết quả mong đợi), và tìm kiếm Biểu diễn tốt nhất trong một Không gian giả thuyết (Hypothesis space) đã được định nghĩa trước.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Câu 12: \"Học\" (Learning) trong bối cảnh học máy được hiểu một cách chính xác nhất là quá trình nào sau đây?",
    options: [
      "Quá trình tìm kiếm tự động các phép biến đổi dữ liệu phù hợp trong một không gian giả thuyết xác định.",
      "Quá trình ghi nhớ toàn bộ cơ sở dữ liệu để có thể truy xuất thông tin ngay lập tức.",
      "Quá trình thiết lập kết nối internet tới các máy chủ toàn cầu để tải xuống các quy tắc.",
      "Quá trình một kỹ sư tự tay hiệu chỉnh các tham số dựa trên suy luận toán học."
    ],
    correctAnswer: 0,
    explanation: "Học tập (learning) là quá trình tìm kiếm tự động các biểu diễn dữ liệu tốt (phép biến đổi dữ liệu) thông qua không gian giả thuyết (các phương pháp đã được xác định).",
    difficulty: "Khó"
  },

  // Topic 5: Sự "sâu" trong "học sâu"
  {
    type: "mcq",
    question: "Câu 13: Chữ \"sâu\" (deep) trong \"học sâu\" (deep learning) thực chất ám chỉ điều gì?",
    options: [
      "Số lượng các lớp biểu diễn kế tiếp nhau được sử dụng trong mô hình.",
      "Sự hiểu biết sâu sắc và phức tạp của AI về ý nghĩa tự nhiên của dữ liệu.",
      "Cấu trúc não bộ con người được mô phỏng chi tiết sâu bên trong thuật toán.",
      "Sự sâu chuỗi của nhiều thuật toán học máy khác nhau vào một bộ xử lý."
    ],
    correctAnswer: 0,
    explanation: "Từ \"sâu\" không liên quan đến sự hiểu biết sâu sắc nào, mà nó chỉ đơn thuần là ý tưởng sử dụng nhiều lớp (layers) kế tiếp nhau để trích xuất đặc trưng dần dần.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Câu 14: Thuật ngữ nào sau đây thường được dùng để chỉ cấu trúc phân lớp trong deep learning?",
    options: [
      "Mạng lưới thần kinh (Neural networks)",
      "Cây quyết định sâu (Deep decision trees)",
      "Máy vector hỗ trợ lớp (Layered SVM)",
      "Rừng ngẫu nhiên sâu (Deep random forests)"
    ],
    correctAnswer: 0,
    explanation: "Cấu trúc phân lớp này hầu như luôn được triển khai dưới dạng mạng nơ-ron (neural networks), trong đó thuật ngữ 'thần kinh' chỉ mang tính biểu tượng mượn từ sinh học.",
    difficulty: "Dễ"
  },
  {
    type: "fill_blank",
    question: "Câu 15: Điền từ thích hợp: Trong deep learning, các lớp biểu diễn dữ liệu ngày càng phức tạp dần. Lớp ban đầu có thể chỉ nhận dạng đường viền, trong khi các lớp cuối cùng có thể nhận diện được hình ảnh một con _________.",
    blanks: ["chó", "mèo", "động vật", "số", "người"],
    explanation: "Quá trình trích xuất đặc trưng diễn ra theo từng lớp. Lớp thấp (nông) nhận dạng đường nét đơn giản, lớp cao (sâu) tổng hợp các đường nét thành hình thù phức tạp như chữ số, khuôn mặt hoặc động vật.",
    difficulty: "Trung bình"
  },

  // Topic 6: Hiểu cách thức hoạt động của deep learning, qua ba hình
  {
    type: "mcq",
    question: "Câu 16: Trong mạng nơ-ron, chức năng của \"hàm mất mát\" (loss function) hoặc \"hàm mục tiêu\" (objective function) là gì?",
    options: [
      "Đo lường khoảng cách giữa kết quả dự đoán của mô hình và kết quả thực tế mục tiêu.",
      "Đếm số lượng các lớp (layers) cần thiết để mạng nơ-ron đạt được hiệu suất tối ưu.",
      "Mã hóa dữ liệu đầu vào thành ma trận số học trước khi đưa vào mạng lưới.",
      "Tự động tạo ra dữ liệu đào tạo mới để gia tăng độ chính xác của mạng lưới."
    ],
    correctAnswer: 0,
    explanation: "Hàm mất mát tính toán 'điểm số' đánh giá độ lệch giữa những gì mạng dự đoán và những gì thực tế mong muốn (true target). Điểm số này càng thấp càng tốt.",
    difficulty: "Trung bình"
  },
  {
    type: "sorting",
    question: "Câu 17: Sắp xếp các bước theo đúng vòng lặp tối ưu hóa (training loop) của mạng lưới thần kinh:",
    steps: [
      "1. Truyền dữ liệu đầu vào qua mạng để đưa ra dự đoán (Forward pass).",
      "2. Hàm mất mát tính toán sai số giữa dự đoán và kết quả thực tế.",
      "3. Trình tối ưu hóa (Optimizer) sử dụng sai số để tính toán hướng điều chỉnh.",
      "4. Trọng số (Weights) của mạng được cập nhật một chút theo hướng giảm sai số."
    ],
    explanation: "Vòng lặp đào tạo bắt đầu bằng việc dự đoán, sau đó đo lường độ sai lệch (loss). Optimizer (sử dụng Backpropagation) sẽ xác định cách cập nhật các trọng số (weights) để vòng lặp sau sai số sẽ nhỏ hơn.",
    difficulty: "Khó"
  },
  {
    type: "mcq",
    question: "Câu 18: \"Trọng số\" (Weights) trong mạng lưới thần kinh có vai trò tương đương với khái niệm nào sau đây trong quá trình học tập?",
    options: [
      "Kiến thức của mô hình, lưu trữ cách thức chuyển đổi dữ liệu ở mỗi lớp.",
      "Cấu trúc mạng lưới, quyết định số lượng nơ-ron được sử dụng trong mô hình.",
      "Dữ liệu kiểm thử, dùng để đối chiếu độ chính xác của kết quả học tập.",
      "Hàm kích hoạt, xác định ngưỡng kích hoạt của các tế bào thần kinh ảo."
    ],
    correctAnswer: 0,
    explanation: "Sự biến đổi được thực hiện bởi một lớp (layer) được tham số hóa bởi các 'trọng số' của nó. Học tập chính là việc tìm ra bộ trọng số chính xác cho tất cả các lớp.",
    difficulty: "Khó"
  },

  // Topic 7: Điều gì làm cho deep learning trở nên khác biệt
  {
    type: "mcq",
    question: "Câu 19: Đặc điểm nổi bật nhất giúp Học sâu vượt trội so với các phương pháp Học máy cổ điển (như SVM hay Cây quyết định) là gì?",
    options: [
      "Khả năng tự động hóa việc học các biểu diễn dữ liệu (trích xuất đặc trưng) mà không cần sự can thiệp thủ công của con người.",
      "Khả năng chạy trên các phần cứng có cấu hình thấp vì thuật toán của nó đã được đơn giản hóa đi rất nhiều.",
      "Hoạt động dựa trên các quy tắc logic if-else được lập trình tự động thay vì dựa vào xác suất thống kê.",
      "Việc chỉ cần sử dụng một lượng rất nhỏ dữ liệu (few-shot learning) để đạt được độ chính xác tuyệt đối."
    ],
    correctAnswer: 0,
    explanation: "Với các phương pháp cũ (shallow learning), con người phải tự trích xuất đặc trưng (Feature engineering). Học sâu hoàn toàn tự động hóa bước này, học các tính năng song song trong một quá trình duy nhất.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Câu 20: Hai đặc điểm thiết yếu nào đã tạo nên thành công vang dội cho thuật toán Học sâu?",
    options: [
      "Học tăng dần (layer-by-layer) ngày càng phức tạp và tự động học song song tất cả các lớp cùng lúc.",
      "Học không giám sát hoàn toàn tự động và khả năng tự tạo ra dữ liệu đào tạo (GANs).",
      "Ghi nhớ các mẫu dữ liệu ở bộ nhớ ngoài và truy xuất bằng thuật toán tìm kiếm cây (Tree search).",
      "Sử dụng biểu diễn dạng đồ thị (Graph) và loại bỏ hoàn toàn việc sử dụng mạng nơ-ron nhân tạo."
    ],
    correctAnswer: 0,
    explanation: "Deep learning cho phép một mô hình học đồng thời (jointly) tất cả các lớp biểu diễn, thay vì học từng bước rời rạc. Việc cập nhật một thay đổi sẽ tự động tinh chỉnh tất cả các lớp cùng lúc.",
    difficulty: "Trung bình"
  },
  {
    type: "fill_blank",
    question: "Câu 21: Điền từ thích hợp: Kỹ thuật biến đổi dữ liệu đầu vào thành một định dạng dễ xử lý hơn bằng tay (thủ công) được gọi là Feature _________ (Trích xuất đặc trưng). Học sâu loại bỏ hoàn toàn nhu cầu về kỹ thuật này.",
    blanks: ["engineering"],
    explanation: "Feature engineering (kỹ thuật trích xuất đặc trưng) là một rào cản lớn của ML truyền thống, tiêu tốn nhiều thời gian và công sức. Học sâu khắc phục điều này bằng cách tự động học các đặc trưng.",
    difficulty: "Khó"
  },

  // Topic 8: Thời đại của AI sáng tạo (Generative AI)
  {
    type: "mcq",
    question: "Câu 22: Sự kiện phát hành ChatGPT của OpenAI vào cuối năm 2022 được ví như khoảnh khắc nào trong lịch sử công nghệ?",
    options: [
      "Khoảnh khắc Netscape (sự bùng nổ của internet đối với công chúng).",
      "Sự cố Y2K (khoảnh khắc lo sợ tột độ về lỗi hệ thống máy tính).",
      "Bong bóng Dotcom (sự sụp đổ của hàng loạt các công ty công nghệ internet).",
      "Sự ra đời của Bitcoin (mở ra kỷ nguyên của công nghệ chuỗi khối tài chính)."
    ],
    correctAnswer: 0,
    explanation: "Sự ra mắt của ChatGPT đã đánh dấu khoảnh khắc Netscape của AI, thu hút hàng trăm triệu người dùng và đưa sức mạnh của AI tạo sinh đến với công chúng.",
    difficulty: "Dễ"
  },
  {
    type: "mcq",
    question: "Câu 23: Mô hình ngôn ngữ lớn (LLM) dựa trên kiến trúc nào là nền tảng cho ChatGPT?",
    options: [
      "Transformer",
      "Mạng nơ-ron tích chập (CNN)",
      "Máy vector hỗ trợ (SVM)",
      "Mạng nơ-ron bộ nhớ dài ngắn (LSTM)"
    ],
    correctAnswer: 0,
    explanation: "Kiến trúc Transformer, được Google giới thiệu vào năm 2017, là nền tảng của các mô hình ngôn ngữ lớn (Large Language Models) mạnh mẽ nhất hiện nay như GPT.",
    difficulty: "Trung bình"
  },
  {
    type: "matching",
    question: "Câu 24: Ghép nối các ứng dụng AI với nhiệm vụ của chúng trong thời đại Generative AI:",
    pairs: [
      { left: "ChatGPT", right: "Trợ lý ảo hỗ trợ lập trình, tư vấn kịch bản và viết mã." },
      { left: "Midjourney", right: "Tạo hình ảnh và nghệ thuật đồ họa chất lượng cao từ văn bản." },
      { left: "Speech-to-text", right: "Hệ thống nhận dạng và phiên âm giọng nói tự động." }
    ],
    explanation: "AI thế hệ mới tập trung mạnh vào khả năng tạo sinh (Generative), bao gồm văn bản (ChatGPT) và hình ảnh (Midjourney, Stable Diffusion).",
    difficulty: "Dễ"
  },

  // Topic 9: Học sâu đã đạt được những gì cho đến nay
  {
    type: "mcq",
    question: "Câu 25: Học sâu ĐÃ CHƯA đạt được thành tựu nào dưới đây (tính đến thời điểm hiện tại của cuốn sách)?",
    options: [
      "Trí tuệ nhân tạo tổng hợp đa năng (AGI - Artificial General Intelligence) có tri giác.",
      "Đạt độ chính xác cấp độ con người trong nhận dạng giọng nói và nhận dạng hình ảnh.",
      "Đánh bại nhà vô địch thế giới của con người trong các trò chơi chiến lược phức tạp như cờ vây (Go).",
      "Dịch văn bản giữa nhiều ngôn ngữ với độ chính xác cao tiếp cận mức độ của con người."
    ],
    correctAnswer: 0,
    explanation: "Mặc dù đạt được vô số thành tựu trong các tác vụ hẹp, AI vẫn chưa đạt đến mức độ AGI (trí thông minh có thể hiểu và tự chủ hoàn toàn như con người).",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Câu 26: Theo tài liệu, lĩnh vực nào đã ghi nhận bước đột phá lịch sử vào năm 2012 nhờ sự xuất hiện của mạng Học sâu?",
    options: [
      "Thị giác máy tính (Computer Vision), đặc biệt là cuộc thi phân loại hình ảnh ImageNet.",
      "Chẩn đoán y tế, khi AI lần đầu tiên thay thế hoàn toàn bác sĩ trong phòng phẫu thuật.",
      "Công nghệ ô tô bay, bằng cách sử dụng AI để điều phối giao thông trên không gian.",
      "Hệ thống định vị GPS không gian sâu, điều hướng tàu vũ trụ lên sao Hỏa tự động."
    ],
    correctAnswer: 0,
    explanation: "Năm 2012, mạng nơ-ron sâu AlexNet đã chiến thắng vang dội trong cuộc thi ImageNet, đánh dấu sự bùng nổ của Deep Learning trong thị giác máy tính.",
    difficulty: "Khó"
  },
  {
    type: "sorting",
    question: "Câu 27: Sắp xếp các mốc thời gian quan trọng trong sự phát triển của AI:",
    steps: [
      "1. Những năm 1950: AI biểu tượng ra đời (Kỳ vọng máy tính có thể lập luận).",
      "2. Những năm 1980: Trào lưu Hệ chuyên gia (Expert systems) bùng nổ rồi suy thoái.",
      "3. Những năm 1990: Máy học (Machine Learning) dần khẳng định vị thế nhờ phần cứng và dữ liệu.",
      "4. Năm 2012: Học sâu (Deep Learning) bứt phá với kỷ lục tại cuộc thi ImageNet."
    ],
    explanation: "Đây là trục thời gian tiêu chuẩn mô tả sự dịch chuyển từ Symbolic AI sang Machine Learning và cuối cùng là sự trỗi dậy của Deep Learning hiện đại.",
    difficulty: "Khó"
  },

  // Topic 10: Tương lai và Lời hứa của AI (Hype vs Reality)
  {
    type: "mcq",
    question: "Câu 28: Theo nhận định của các chuyên gia trong cuốn sách, \"Mùa đông AI\" (AI Winter) là gì?",
    options: [
      "Giai đoạn mà nguồn tài trợ và sự quan tâm dành cho AI bị cắt giảm nghiêm trọng do những kỳ vọng không thực tế không được đáp ứng.",
      "Tình trạng các siêu máy tính bị quá nhiệt và phải tạm ngưng hoạt động vào mùa đông để tiết kiệm chi phí làm mát.",
      "Một thuật toán học máy chỉ có thể hoạt động tối ưu trong điều kiện nhiệt độ phòng thí nghiệm thấp để giảm nhiễu.",
      "Giai đoạn các nhà nghiên cứu AI nghỉ ngơi và tập trung vào các lĩnh vực phần mềm khác do đã cạn ý tưởng thuật toán."
    ],
    correctAnswer: 0,
    explanation: "AI Winter xảy ra khi sự cường điệu (hype) quá lớn nhưng công nghệ không đáp ứng kịp, dẫn đến sự thất vọng và rút vốn hàng loạt của các nhà đầu tư.",
    difficulty: "Trung bình"
  },
  {
    type: "mcq",
    question: "Câu 29: Thái độ nào được khuyên dùng nhất khi đối mặt với sự cường điệu (hype) về AI hiện nay?",
    options: [
      "Đừng tin vào sự cường điệu ngắn hạn, nhưng hãy tin tưởng vào tầm nhìn và tiềm năng biến đổi của AI trong dài hạn.",
      "Chấp nhận hoàn toàn mọi lời hứa hẹn của truyền thông vì AI hiện nay không có giới hạn nào cả.",
      "Bỏ qua mọi tin tức về AI và chỉ sử dụng các thuật toán truyền thống để đảm bảo độ tin cậy tuyệt đối.",
      "Ngừng phát triển AI lập tức vì nó đe dọa trực tiếp đến sự tồn vong của toàn bộ lao động con người."
    ],
    correctAnswer: 0,
    explanation: "Truyền thông thường cường điệu hóa về AI trong ngắn hạn (như việc AI có tri giác hay thay thế hoàn toàn con người ngay lập tức). Chúng ta cần giữ cái nhìn thực tế trong ngắn hạn nhưng lạc quan về tiềm năng dài hạn.",
    difficulty: "Dễ"
  },
  {
    type: "fill_blank",
    question: "Câu 30: Điền từ thích hợp: Ba rào cản lớn nhất ngăn cản Học sâu thành công trước thập niên 2010 là: Thuật toán hạn chế, thiếu Dữ liệu quy mô lớn và sự yếu kém của _________ (Hardware).",
    blanks: ["phần cứng", "máy tính", "phần cứng máy tính"],
    explanation: "Học sâu yêu cầu sức mạnh tính toán khổng lồ (đặc biệt là GPU) và tập dữ liệu lớn. Cho đến những năm 2010, phần cứng (đặc biệt là đồ họa) và internet mới cung cấp đủ hai yếu tố này.",
    difficulty: "Trung bình"
  }
];

export default quizData;
