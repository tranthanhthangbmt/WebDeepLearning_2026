const quizData = [
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 1: Thuật ngữ 'Target' (Mục tiêu) trong học máy (machine learning) được hiểu là gì?",
    options: [
      "Một điểm dữ liệu đầu vào duy nhất đưa vào mô hình để tính toán.",
      "Thước đo khoảng cách giữa dự đoán của mô hình và nhãn thực tế.",
      "Sự thật (ground-truth), tức là giá trị mà mô hình lý tưởng nên dự đoán được.",
      "Tập hợp các nhãn có thể được chọn trong bài toán phân loại đa lớp."
    ],
    correctAnswer: 2,
    explanation: "Target là sự thật hay nhãn chính xác do con người gán (ground-truth). Nó là mục tiêu mà mô hình cần phải dự đoán được dựa trên dữ liệu đầu vào."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 2: Ví dụ nào sau đây mô tả bài toán phân loại nhị phân (binary classification)?",
    options: [
      "Dự đoán giá của một ngôi nhà dựa trên diện tích và số phòng ngủ.",
      "Phân loại các bài báo tức thành 46 chủ đề tin tức khác nhau.",
      "Nhận diện xem một email có phải là thư rác (spam) hay không.",
      "Gán nhiều nhãn cho một bức ảnh (ví dụ: vừa có 'chó' vừa có 'mèo')."
    ],
    correctAnswer: 2,
    explanation: "Phân loại nhị phân chỉ có hai kết quả loại trừ lẫn nhau. Ví dụ: Thư rác hoặc Không phải thư rác, Đánh giá phim Tích cực hoặc Tiêu cực."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 3: Trong học sâu, thuật ngữ 'Loss value' (Giá trị mất mát) dùng để đo lường điều gì?",
    options: [
      "Tốc độ xử lý của phần cứng (GPU) khi chạy thuật toán.",
      "Khoảng cách giữa dự đoán của mô hình so với mục tiêu thực tế.",
      "Tổng số lượng trọng số (weights) của một mạng nơ-ron sâu.",
      "Số lượng các nhãn (labels) bị thiếu trong tập dữ liệu huấn luyện."
    ],
    correctAnswer: 1,
    explanation: "Loss value (Giá trị mất mát) đại diện cho mức độ sai lệch của mô hình. Trong quá trình huấn luyện, mục tiêu của thuật toán tối ưu (optimizer) là làm giảm giá trị này xuống mức thấp nhất."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 4: Dự đoán giá của một ngôi nhà dựa trên các thông số bất động sản là ví dụ của loại bài toán nào?",
    options: [
      "Phân loại nhiều nhãn (Multilabel classification)",
      "Hồi quy vector (Vector regression)",
      "Phân loại nhị phân (Binary classification)",
      "Hồi quy vô hướng (Scalar regression)"
    ],
    correctAnswer: 3,
    explanation: "Vì giá nhà là một giá trị số liên tục và chỉ có một giá trị duy nhất (vô hướng), bài toán này thuộc loại Scalar regression."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 5: Tại sao trong bài toán phân loại đánh giá phim IMDb, chúng ta lại giới hạn 'num_words=10000'?",
    options: [
      "Vì mạng nơ-ron chỉ có khả năng tính toán ma trận với số cột đúng bằng 10,000.",
      "Để tập trung vào các từ xuất hiện thường xuyên và loại bỏ các từ quá hiếm gặp.",
      "Vì tập dữ liệu gốc chỉ chứa đúng 10,000 từ tiếng Anh phân biệt (unique words).",
      "Để giới hạn số lượng bài đánh giá phim tải về máy nhằm tiết kiệm băng thông."
    ],
    correctAnswer: 1,
    explanation: "Việc loại bỏ các từ hiếm (chỉ xuất hiện một vài lần) giúp giảm kích thước không gian vector dữ liệu, tập trung tài nguyên mô hình vào những từ thực sự mang tính phân loại."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 6: Kỹ thuật biến đổi một danh sách các chỉ số từ thành một vector bao gồm các số 0 và 1 (để biểu diễn sự xuất hiện của từ) được gọi là mã hóa ________.",
    blanks: ["one-hot", "multi-hot", "one hot", "multi hot"],
    explanation: "Mã hóa Multi-hot (hoặc one-hot) chuyển chuỗi các số nguyên thành một vector có độ dài cố định (ví dụ 10,000 chiều), trong đó các vị trí tương ứng với từ xuất hiện sẽ được gán giá trị 1, phần còn lại là 0."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 7: Nếu một lớp (layer) không sử dụng hàm kích hoạt (activation function) như 'relu', điều gì sẽ xảy ra?",
    options: [
      "Mô hình sẽ sinh ra các giá trị ngẫu nhiên và không thể hội tụ được.",
      "Lớp đó sẽ chỉ thực hiện các phép biến đổi tuyến tính (affine transformations).",
      "Kích thước bộ nhớ cần thiết để chạy mô hình sẽ tăng lên gấp đôi bình thường.",
      "Lớp đó sẽ tự động bị Keras vô hiệu hóa để bảo vệ kiến trúc mô hình."
    ],
    correctAnswer: 1,
    explanation: "Nếu không có hàm phi tuyến (non-linearity) như relu, việc xếp chồng nhiều lớp Dense thực chất cũng chỉ tương đương với một phép biến đổi tuyến tính duy nhất. Hàm kích hoạt giúp mạng nơ-ron học được không gian giả thuyết phức tạp hơn."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 8: Tác dụng chính của hàm kích hoạt 'relu' (Rectified Linear Unit) là gì?",
    options: [
      "Nén tất cả các giá trị đầu vào để chúng nằm trong khoảng từ 0 đến 1.",
      "Chuyển toàn bộ các giá trị âm thành số 0 và giữ nguyên giá trị dương.",
      "Tính trung bình cộng của tất cả các trọng số (weights) trong một lớp.",
      "Chuyển đổi dữ liệu chuỗi văn bản thành dữ liệu số nguyên (integer sequence)."
    ],
    correctAnswer: 1,
    explanation: "Đồ thị của Relu là một đường gấp khúc: max(x, 0). Nó loại bỏ hoàn toàn các giá trị âm (trả về 0) và giữ nguyên các giá trị dương, tạo ra tính phi tuyến cho mô hình."
  },
  {
    type: "sorting",
    difficulty: "Khó",
    question: "Câu 9: Sắp xếp các bước chuẩn bị dữ liệu văn bản (như IMDb) trước khi đưa vào mạng Dense:",
    steps: [
      "Tải dữ liệu dưới dạng danh sách các số nguyên (chỉ số của từ điển).",
      "Sử dụng kỹ thuật multi-hot encoding để tạo mảng vector chứa toàn số 0.",
      "Đánh dấu số 1 tại các vị trí chỉ số tương ứng với từ xuất hiện trong câu.",
      "Chuyển đổi mảng nhãn (labels) từ kiểu số nguyên sang kiểu số thực (float32)."
    ],
    explanation: "Để đưa văn bản vào mạng Dense, dữ liệu chuỗi số nguyên phải được mã hóa thành các vector nhị phân cùng độ dài thông qua multi-hot encoding, sau đó chuẩn hóa nhãn."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 10: Trong bài toán phân loại nhị phân (Binary classification), lớp Dense cuối cùng thường có cấu trúc nào?",
    options: [
      "layers.Dense(2, activation='relu')",
      "layers.Dense(1, activation='sigmoid')",
      "layers.Dense(2, activation='softmax')",
      "layers.Dense(1, activation='linear')"
    ],
    correctAnswer: 1,
    explanation: "Mô hình phân loại nhị phân chỉ cần 1 output unit. Hàm sigmoid sẽ 'bóp' đầu ra về khoảng [0, 1] để đại diện cho xác suất (probability) của lớp 1."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 11: Hàm mất mát (Loss function) nào là lựa chọn tốt nhất cho bài toán phân loại nhị phân (đầu ra là xác suất)?",
    options: [
      "mean_squared_error",
      "binary_crossentropy",
      "categorical_crossentropy",
      "mean_absolute_error"
    ],
    correctAnswer: 1,
    explanation: "Binary crossentropy đo lường khoảng cách giữa phân phối xác suất dự đoán và phân phối thực tế, cực kỳ hiệu quả khi lớp cuối cùng dùng hàm sigmoid."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 12: Mục đích cốt lõi của việc sử dụng tập dữ liệu xác thực (Validation set) là gì?",
    options: [
      "Đóng vai trò thay thế nếu tập dữ liệu huấn luyện (Training set) bị mất.",
      "Theo dõi hiệu suất của mô hình trên dữ liệu mới trong quá trình huấn luyện.",
      "Dùng để tính toán trực tiếp giá trị gradient nhằm cập nhật trọng số.",
      "Giúp tăng kích thước (dimensionality) của các mẫu dữ liệu đầu vào."
    ],
    correctAnswer: 1,
    explanation: "Tập xác thực (Validation set) không được dùng để tính gradient. Nó đóng vai trò làm 'giám khảo' độc lập, giúp phát hiện sớm hiện tượng Overfitting khi mô hình học vẹt dữ liệu huấn luyện."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 13: Hiện tượng Overfitting (Quá khớp) xảy ra khi nào?",
    options: [
      "Khi mô hình không học được gì từ tập dữ liệu huấn luyện và có Loss rất cao.",
      "Khi mô hình hoạt động tốt trên tập huấn luyện nhưng lại rất kém trên dữ liệu mới.",
      "Khi kích thước batch size được thiết lập quá lớn (ví dụ: trên 10,000 samples).",
      "Khi chúng ta sử dụng quá ít số lượng epochs trong quá trình huấn luyện."
    ],
    correctAnswer: 1,
    explanation: "Overfitting là khi mô hình ghi nhớ các đặc điểm nhiễu (noise) của tập huấn luyện. Biểu hiện là Training Loss tiếp tục giảm nhưng Validation Loss lại tăng lên."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 14: Trong hàm `model.fit()`, đối số giúp tự động trích xuất một phần dữ liệu huấn luyện để làm tập xác thực là ________.",
    blanks: ["validation_split", "validation split"],
    explanation: "Bằng cách khai báo `validation_split=0.2`, Keras sẽ tự động trích ra 20% dữ liệu huấn luyện cuối cùng để dùng làm tập xác thực (validation data)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 15: Trong bài toán phân loại đa lớp (Multiclass classification) như phân loại bản tin Reuters, cấu trúc lớp cuối cùng nên là gì?",
    options: [
      "layers.Dense(46, activation='relu')",
      "layers.Dense(1, activation='sigmoid')",
      "layers.Dense(46, activation='softmax')",
      "layers.Dense(1, activation='softmax')"
    ],
    correctAnswer: 2,
    explanation: "Vì có 46 chủ đề, ta cần lớp Dense có 46 đơn vị. Hàm Softmax sẽ chuyển đổi 46 giá trị đầu ra thành một phân phối xác suất (tổng bằng 1)."
  },
  {
    type: "matching",
    difficulty: "Khó",
    question: "Câu 16: Ghép nối tên bài toán với Cấu trúc lớp cuối & Hàm mất mát lý tưởng nhất:",
    pairs: [
      { left: "Phân loại nhị phân (Binary classification)", right: "Dense(1, activation='sigmoid') + binary_crossentropy" },
      { left: "Phân loại đa lớp, nhãn one-hot (Multiclass)", right: "Dense(N, activation='softmax') + categorical_crossentropy" },
      { left: "Phân loại đa lớp, nhãn là số nguyên", right: "Dense(N, activation='softmax') + sparse_categorical_crossentropy" },
      { left: "Hồi quy vô hướng (Scalar regression)", right: "Dense(1, activation=None) + mean_squared_error" }
    ],
    explanation: "Đây là công thức thiết kế kiến trúc chuẩn cho 3 bài toán phổ biến nhất trong Học sâu."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 17: Thông tin bị thắt nút cổ chai (Information bottleneck) trong mạng nơ-ron đa lớp xảy ra vì nguyên nhân nào?",
    options: [
      "Sử dụng tốc độ học (learning rate) quá nhỏ trong quá trình tối ưu.",
      "Sử dụng một lớp trung gian có số lượng đơn vị (units) quá nhỏ so với số lớp đầu ra.",
      "Truyền dữ liệu bằng tensor 1D thay vì sử dụng tensor 3D.",
      "Sử dụng quá nhiều epoch khiến bộ nhớ đệm (cache) bị tràn."
    ],
    correctAnswer: 1,
    explanation: "Nếu bạn dự đoán 46 lớp ở đầu ra nhưng lại có một lớp trung gian chỉ có 4 units, dữ liệu sẽ bị ép phải nén quá mức, gây mất mát thông tin vĩnh viễn không thể phục hồi ở các lớp sau."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 18: Kỹ thuật 'One-hot encoding' áp dụng cho nhãn (labels) trong phân loại đa lớp có chức năng gì?",
    options: [
      "Gom nhóm các nhãn giống nhau thành một nhãn duy nhất để tính toán nhanh hơn.",
      "Biến nhãn nguyên (ví dụ: 3) thành vector chứa toàn bộ số 0, trừ vị trí số 3 là số 1.",
      "Đảo ngược trình tự của các từ trong văn bản để tăng cường dữ liệu (data augmentation).",
      "Chuẩn hóa dữ liệu đưa vào bằng cách trừ đi giá trị trung bình."
    ],
    correctAnswer: 1,
    explanation: "One-hot encoding cho nhãn mục tiêu (categorical encoding) là định dạng cần thiết để tính hàm mất mát `categorical_crossentropy`. Bạn có thể dùng `keras.utils.to_categorical` để thực hiện việc này."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 19: Thuộc tính `history` trả về từ hàm `model.fit()` chứa thông tin gì?",
    options: [
      "Bản sao lưu trữ toàn bộ tập dữ liệu (dataset) để tái sử dụng sau này.",
      "Chứa thông tin cấu trúc mạng (số lớp, số node) để hiển thị biểu đồ.",
      "Một từ điển lưu trữ lại lịch sử các chỉ số như loss và accuracy qua từng epoch.",
      "Mã nguồn nội bộ (source code) của trình tối ưu hóa đã được sử dụng."
    ],
    correctAnswer: 2,
    explanation: "Đối tượng history.history là một dictionary ghi lại mọi số liệu đo lường (loss, val_loss, accuracy, val_accuracy) ở cuối mỗi epoch. Nó rất hữu ích để vẽ biểu đồ đánh giá Overfitting."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 20: Tại sao trong bài toán Hồi quy (Regression) dự đoán giá nhà, lớp cuối cùng KHÔNG sử dụng hàm kích hoạt nào?",
    options: [
      "Vì sử dụng hàm kích hoạt sẽ làm tăng đáng kể chi phí tính toán.",
      "Để mạng có thể tự do dự đoán ra giá trị nằm ở bất kỳ phạm vi tuyến tính nào.",
      "Vì dữ liệu đầu vào đã được chuẩn hóa (normalized) từ trước.",
      "Vì Keras mặc định đã tự động áp dụng hàm sigmoid cho lớp cuối cùng."
    ],
    correctAnswer: 1,
    explanation: "Trong Hồi quy vô hướng, ta dự đoán một số thực. Nếu dùng Sigmoid, kết quả sẽ bị giới hạn ở [0, 1]. Việc không dùng hàm kích hoạt (linear layer) cho phép mạng dự đoán các giá trị số thực dương và âm tùy ý."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 21: Thước đo đánh giá thường được sử dụng thay thế cho Accuracy trong bài toán Hồi quy là Sai số tuyệt đối trung bình, viết tắt tiếng Anh là ________.",
    blanks: ["MAE", "mae", "Mean Absolute Error"],
    explanation: "Trong bài toán hồi quy (như dự đoán giá nhà), Accuracy không có ý nghĩa. Người ta dùng MAE (Mean Absolute Error) để biết mô hình dự đoán sai lệch bao nhiêu đơn vị so với thực tế."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 22: Khi dữ liệu huấn luyện (training data) quá ít, tại sao người ta lại sử dụng kỹ thuật K-fold cross-validation thay vì chia Validation set thông thường?",
    options: [
      "Vì tập Validation nhỏ sẽ khiến kết quả đánh giá (validation score) biến động quá mạnh.",
      "Vì hàm mất mát sẽ bị lỗi chia cho 0 nếu tập xác thực có ít hơn 100 mẫu.",
      "Vì GPU không thể xử lý các tập xác thực có kích thước lẻ không phải lũy thừa của 2.",
      "Vì K-fold giúp mô hình tránh bị giới hạn phần cứng khi chạy trên CPU."
    ],
    correctAnswer: 0,
    explanation: "Khi dữ liệu rất ít, việc tách một phần làm tập validation sẽ dẫn đến sự dao động (variance) cực lớn trong kết quả đánh giá tùy thuộc vào việc mẫu nào bị tách ra. K-fold chia dữ liệu làm K phần và huấn luyện K mô hình khác nhau để lấy điểm trung bình, giúp đánh giá chính xác và ổn định hơn."
  },
  {
    type: "sorting",
    difficulty: "Trung bình",
    question: "Câu 23: Quy trình chuẩn hóa dữ liệu đầu vào (Feature-wise Normalization) trong bài toán dự đoán giá nhà Boston:",
    steps: [
      "Tính toán giá trị Trung bình (Mean) của từng đặc trưng trên tập dữ liệu huấn luyện.",
      "Tính toán Độ lệch chuẩn (Standard Deviation) của từng đặc trưng.",
      "Trừ mỗi giá trị đầu vào cho giá trị Trung bình tương ứng.",
      "Chia kết quả vừa tính cho Độ lệch chuẩn tương ứng."
    ],
    explanation: "Quy trình Feature Normalization: `(x - mean) / std`. LƯU Ý: Mean và Std phải luôn được tính trên tập Huấn luyện (Training set), sau đó áp dụng con số đó cho cả tập Validation và tập Test để tránh rò rỉ dữ liệu."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 24: Điều gì là QUAN TRỌNG NHẤT cần nhớ khi chuẩn hóa dữ liệu kiểm thử (test data) trong bài toán học máy?",
    options: [
      "Tuyệt đối không bao giờ được chuẩn hóa dữ liệu kiểm thử.",
      "Chỉ sử dụng Mean và Standard Deviation được tính toán từ tập dữ liệu huấn luyện.",
      "Phải tính toán Mean và Standard Deviation trên chính tập dữ liệu kiểm thử đó.",
      "Dữ liệu kiểm thử phải được làm tròn thành số nguyên trước khi chuẩn hóa."
    ],
    correctAnswer: 1,
    explanation: "Đây là nguyên tắc 'Không rò rỉ dữ liệu' (No Data Leakage). Các đại lượng chuẩn hóa (Mean, Std) phải luôn được lấy từ tập Training. Không được dùng bất kỳ thông tin gì từ tập Test vào quy trình chuẩn hóa."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 25: Trong Keras, sự khác biệt giữa hàm `model.fit()` và `model.evaluate()` là gì?",
    options: [
      "fit() dùng để tải tập dữ liệu, evaluate() dùng để xóa dữ liệu khỏi bộ nhớ.",
      "fit() sẽ cập nhật trọng số để mô hình học, evaluate() chỉ tính điểm hiệu suất.",
      "fit() chỉ chạy trên CPU, evaluate() bắt buộc phải chạy trên môi trường GPU.",
      "fit() áp dụng cho mạng nhiều lớp, evaluate() dùng cho mạng có một lớp duy nhất."
    ],
    correctAnswer: 1,
    explanation: "Hàm `fit()` kích hoạt cơ chế Backpropagation để tối ưu hóa trọng số. Ngược lại, `evaluate()` chỉ chạy mô hình ở chế độ inference (dự đoán) để đo lường Loss và Metrics trên tập Test."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 26: Bạn có một tập dữ liệu phân loại với nhãn được đánh số nguyên từ 0 đến 45. Nếu không muốn mã hóa One-hot cho nhãn, bạn phải dùng hàm mất mát nào?",
    options: [
      "mean_squared_error",
      "binary_crossentropy",
      "sparse_categorical_crossentropy",
      "categorical_crossentropy"
    ],
    correctAnswer: 2,
    explanation: "Nếu nhãn là mảng số nguyên (ví dụ: `[3, 45, 12]`), thay vì mảng One-hot, Keras cung cấp `sparse_categorical_crossentropy` để tính toán mất mát. Nó hoàn toàn tương đương về mặt toán học với `categorical_crossentropy`."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 27: Khi huấn luyện mạng nơ-ron trên một tập dữ liệu cực kỳ nhỏ, phương pháp thiết kế mô hình nào là tốt nhất để giảm thiểu overfitting?",
    options: [
      "Sử dụng mô hình cực sâu với hàng trăm lớp (layers) để học tốt hơn.",
      "Sử dụng mạng rất nhỏ (ít lớp và ít units) để hạn chế dung lượng ghi nhớ.",
      "Thay đổi hàm kích hoạt từ relu sang hàm tuyến tính thông thường.",
      "Huấn luyện mô hình trong hàng triệu epochs liên tục không dừng."
    ],
    correctAnswer: 1,
    explanation: "Dữ liệu nhỏ rất dễ gây ra Overfitting. Mạng lưới càng lớn (nhiều tham số) thì khả năng 'ghi nhớ máy móc' dữ liệu càng mạnh. Do đó, giảm kích thước mạng là cách phòng vệ đầu tiên."
  },
  {
    type: "matching",
    difficulty: "Dễ",
    question: "Câu 28: Ghép nối các thuật ngữ học máy với ý nghĩa đúng của nó:",
    pairs: [
      { left: "Sample (Mẫu)", right: "Một điểm dữ liệu duy nhất đưa vào mô hình" },
      { left: "Batch (Lô)", right: "Một tập hợp nhỏ các mẫu được xử lý cùng lúc" },
      { left: "Epoch", right: "Một vòng lặp huấn luyện đi qua toàn bộ tập dữ liệu" },
      { left: "Label (Nhãn)", right: "Định danh cụ thể của lớp mà mô hình cần phân loại" }
    ],
    explanation: "Đây là bảng chú giải thuật ngữ cơ bản (Glossary) xuất hiện xuyên suốt mọi tài liệu Deep Learning."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 29: Nếu bài toán yêu cầu gán nhiều nhãn cho một hình ảnh (ví dụ: một bức ảnh có cả chó và mèo), kiến trúc đầu ra nào là chính xác nhất?",
    options: [
      "Lớp Dense có N đơn vị kết hợp cùng hàm kích hoạt softmax.",
      "Lớp Dense có N đơn vị kết hợp cùng hàm kích hoạt sigmoid.",
      "Lớp Dense có 1 đơn vị kết hợp cùng hàm kích hoạt sigmoid.",
      "Lớp Dense có N đơn vị và không sử dụng hàm kích hoạt."
    ],
    correctAnswer: 1,
    explanation: "Trong Multilabel classification, mỗi lớp (class) là độc lập với nhau (xác suất có chó không ảnh hưởng đến xác suất có mèo). Vì vậy, ta phải dùng N hàm Sigmoid độc lập thay vì Softmax (vốn ép tổng các xác suất bằng 1)."
  },
  {
    type: "fill_blank",
    difficulty: "Dễ",
    question: "Câu 30: Phương thức dùng để sử dụng một mô hình đã huấn luyện xong vào việc tạo ra dự đoán trên dữ liệu mới trong Keras là `model.________`.",
    blanks: ["predict", "predict()", "predict(x)"],
    explanation: "Hàm `model.predict(new_data)` trả về kết quả dự đoán của mô hình. Tùy thuộc cấu trúc mạng, nó có thể trả về xác suất, nhãn phân loại hoặc một con số liên tục."
  }
];

export default quizData;
