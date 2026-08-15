const quizData = [
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 1: Vấn đề cơ bản và trung tâm nhất trong học máy (machine learning) là gì?",
    options: [
      "Sự thiếu hụt nguồn tài nguyên GPU để huấn luyện mạng nơ-ron sâu.",
      "Làm sao để tìm được một bộ dữ liệu hoàn toàn không có bất kỳ nhiễu nào.",
      "Sự căng thẳng giữa việc tối ưu hóa mô hình và khả năng khái quát hóa.",
      "Làm sao để người dùng hiểu được cấu trúc ẩn bên trong mạng nơ-ron."
    ],
    correctAnswer: 2,
    explanation: "Cốt lõi của học máy là sự cân bằng giữa tối ưu hóa (học tốt trên dữ liệu huấn luyện) và khái quát hóa (dự đoán tốt trên dữ liệu mới chưa từng thấy)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 2: 'Tối ưu hóa' (Optimization) trong bối cảnh huấn luyện mô hình được hiểu là gì?",
    options: [
      "Việc nén kích thước mô hình để nó chạy nhanh hơn trên thiết bị di động.",
      "Quá trình điều chỉnh mô hình để đạt hiệu suất tốt nhất trên dữ liệu huấn luyện.",
      "Kỹ thuật loại bỏ các đặc trưng (features) không cần thiết khỏi tập dữ liệu.",
      "Mức độ hoạt động tốt của mô hình khi gặp một tập dữ liệu hoàn toàn mới."
    ],
    correctAnswer: 1,
    explanation: "Tối ưu hóa chính là quá trình 'học' của máy, trong đó mô hình liên tục cập nhật trọng số để giảm thiểu sai số trên tập dữ liệu dùng để huấn luyện."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 3: Hiện tượng mô hình hoạt động cực kỳ xuất sắc trên tập huấn luyện nhưng lại rất tệ trên dữ liệu mới gọi là gì?",
    options: [
      "Trang bị thiếu (Underfitting)",
      "Khái quát hóa (Generalization)",
      "Trang bị quá mức (Overfitting)",
      "Chuẩn hóa dữ liệu (Normalization)"
    ],
    correctAnswer: 2,
    explanation: "Overfitting (Trang bị quá mức) xảy ra khi mô hình bắt đầu 'học vẹt' các chi tiết nhiễu hoặc các mẫu cục bộ của tập dữ liệu huấn luyện, khiến nó mất đi khả năng khái quát."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 4: Khi cả tổn thất trên tập huấn luyện (training loss) và tổn thất trên tập kiểm tra (test loss) đều đang giảm, mô hình đang ở trạng thái nào?",
    options: [
      "Mô hình đang ở trạng thái trang bị quá mức (Overfitting).",
      "Mô hình đã đạt được mức khái quát hóa hoàn hảo nhất.",
      "Mô hình đang bị thiếu dung lượng bộ nhớ để tiếp tục học.",
      "Mô hình đang ở trạng thái trang bị thiếu (Underfitting)."
    ],
    correctAnswer: 3,
    explanation: "Khi test loss vẫn còn tiếp tục giảm song song với training loss, nghĩa là mô hình vẫn chưa học hết các mẫu quan trọng. Trạng thái này gọi là Underfitting (Trang bị thiếu)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 5: Nguyên nhân nào sau đây KHÔNG phải là lý do phổ biến dẫn đến Overfitting?",
    options: [
      "Tập dữ liệu huấn luyện chứa nhiều nhãn bị gán sai.",
      "Mô hình có quá nhiều tham số (quá lớn) so với lượng dữ liệu.",
      "Tập dữ liệu có quá ít điểm dữ liệu bao phủ không gian đặc trưng.",
      "Mô hình có kiến trúc quá nhỏ, chỉ gồm một lớp với vài nơ-ron."
    ],
    correctAnswer: 3,
    explanation: "Một mô hình quá nhỏ (ít tham số) sẽ không có đủ dung lượng (capacity) để ghi nhớ dữ liệu. Do đó, nó có xu hướng bị Underfitting chứ không thể bị Overfitting."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 6: Tại sao các tính năng mơ hồ (ambiguous features) lại gây khó khăn cho việc khái quát hóa?",
    options: [
      "Vì chúng làm cho tốc độ tính toán gradient trở nên chậm đi đáng kể.",
      "Vì chúng có thể khiến mô hình tự tin thái quá vào các vùng ranh giới.",
      "Vì chúng buộc các lớp mạng nơ-ron phải chuyển sang dùng hàm tuyến tính.",
      "Vì chúng tự động xóa bỏ những trọng số đã được cập nhật trước đó."
    ],
    correctAnswer: 1,
    explanation: "Những vùng dữ liệu mơ hồ (như ảnh quả chuối vừa xanh vừa chín) nếu bị mô hình cố tình học thuộc và phân loại với độ tự tin quá cao sẽ dẫn đến overfitting, thay vì học cách bỏ qua chúng."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 7: Việc một từ hiếm gặp ngẫu nhiên xuất hiện nhiều trong các bài đánh giá tiêu cực và khiến mô hình đánh giá sai trong tương lai được gọi là hiện tượng tương quan ________.",
    blanks: ["giả", "giả mạo", "spurious", "spurious correlation"],
    explanation: "Spurious correlation (Tương quan giả) là hiện tượng mô hình học được những mối liên hệ ngẫu nhiên trong dữ liệu huấn luyện nhưng hoàn toàn không đúng trong thực tế."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 8: Thí nghiệm chèn thêm kênh nhiễu trắng (white noise channels) vào dữ liệu MNIST đã chứng minh điều gì?",
    options: [
      "Mạng nơ-ron không thể huấn luyện được nếu dữ liệu có chứa nhiễu.",
      "Các đặc trưng nhiễu luôn làm giảm nhẹ độ chính xác xác thực của mô hình.",
      "Việc chèn thêm chiều dữ liệu mới sẽ luôn giúp tăng khả năng khái quát hóa.",
      "Kênh nhiễu trắng giúp mô hình học nhanh hơn nhờ sự khởi tạo ngẫu nhiên."
    ],
    correctAnswer: 1,
    explanation: "Thí nghiệm cho thấy dù thông tin gốc không đổi, việc thêm nhiễu sẽ tạo ra các 'tương quan giả'. Mô hình sẽ bị phân tâm bởi nhiễu này và kết quả là Validation Accuracy bị giảm sút."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 9: Giả thuyết đa tạp (Manifold hypothesis) trong học sâu phát biểu điều gì?",
    options: [
      "Mọi dữ liệu thế giới thực luôn tồn tại dưới dạng các mảng đa chiều ngẫu nhiên.",
      "Mạng nơ-ron đa lớp có khả năng học được mọi hàm toán học tuyến tính.",
      "Dữ liệu tự nhiên nằm trên một không gian con (đa tạp) có số chiều thấp.",
      "Để giải quyết vấn đề phân loại, cần đưa dữ liệu về dạng một mặt cầu 3D."
    ],
    correctAnswer: 2,
    explanation: "Manifold hypothesis cho rằng dù dữ liệu đầu vào (như ảnh) có số chiều khổng lồ, những mẫu dữ liệu hợp lệ thực chất chỉ nằm trên một cấu trúc (đa tạp) có số chiều rất thấp và có tính liên tục."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 10: Khả năng khái quát hóa của Học sâu chủ yếu phụ thuộc vào cơ chế toán học nào?",
    options: [
      "Sự suy luận logic (Logic reasoning)",
      "Nội suy (Interpolation) trên đa tạp",
      "Ngoại suy (Extrapolation) tuyến tính",
      "Khởi tạo ngẫu nhiên (Random initialization)"
    ],
    correctAnswer: 1,
    explanation: "Học sâu hoạt động như một công cụ khớp đường cong (curve fitting). Nó dự đoán các dữ liệu chưa từng thấy bằng cách 'nội suy' trơn tru giữa các điểm dữ liệu huấn luyện đã biết nằm trên đa tạp."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 11: Yếu tố nào sau đây là quan trọng NHẤT để mô hình học sâu có thể khái quát hóa tốt?",
    options: [
      "Sử dụng thuật toán tối ưu hóa phức tạp nhất hiện có trên thị trường.",
      "Mô hình phải có số lượng lớp ẩn nhiều gấp đôi số đặc trưng đầu vào.",
      "Tập dữ liệu huấn luyện phải lấy mẫu dày đặc (dense sampling) không gian đầu vào.",
      "Hàm kích hoạt ở tất cả các lớp trung gian bắt buộc phải là hàm sigmoid."
    ],
    correctAnswer: 2,
    explanation: "Vì học sâu dựa vào việc nội suy, không gian đầu vào (đặc biệt là quanh các ranh giới quyết định) phải được lấy mẫu dày đặc. Nếu dữ liệu quá thưa thớt, mô hình không thể nội suy chính xác."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 12: Tại sao chúng ta KHÔNG NÊN đánh giá mô hình cuối cùng bằng Tập dữ liệu xác thực (Validation set)?",
    options: [
      "Vì tập dữ liệu xác thực thường có kích thước quá nhỏ để đưa ra kết luận.",
      "Vì tập xác thực luôn bị nhiễu do quá trình chuẩn hóa dữ liệu đầu vào.",
      "Vì mô hình đã vô tình rò rỉ thông tin từ tập này qua quá trình tinh chỉnh.",
      "Vì tập xác thực chỉ được phép sử dụng duy nhất một lần trong suốt vòng đời."
    ],
    correctAnswer: 2,
    explanation: "Quá trình điều chỉnh siêu tham số (như chọn số epoch, số nơ-ron) dựa trên Validation set sẽ khiến mô hình dần bị 'overfit' vào chính tập này. Do đó, cần một tập Test hoàn toàn độc lập."
  },
  {
    type: "matching",
    difficulty: "Trung bình",
    question: "Câu 13: Ghép nối các chiến lược đánh giá mô hình với đặc điểm đúng của chúng:",
    pairs: [
      { left: "Hold-out validation", right: "Tách một phần dữ liệu cố định làm tập xác thực." },
      { left: "K-fold cross-validation", right: "Chia dữ liệu thành K phần, xoay vòng huấn luyện K lần." },
      { left: "Iterated K-fold validation", right: "Thực hiện K-fold nhiều lần với việc xáo trộn dữ liệu." },
      { left: "Common-sense baseline", right: "Mức hiệu suất tối thiểu của một phương pháp giải quyết ngây thơ." }
    ],
    explanation: "Hold-out dùng khi có nhiều dữ liệu. K-fold dùng khi dữ liệu ít để có kết quả đánh giá ổn định. Iterated K-fold dùng khi dữ liệu cực kỳ ít. Baseline dùng để xác nhận mô hình thực sự đang 'học'."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 14: Khi dự đoán thời tiết ngày mai dựa trên dữ liệu khí tượng trong quá khứ, thao tác nào dưới đây là SAI lầm nghiêm trọng?",
    options: [
      "Bỏ qua các đặc trưng nhiễu không có sự tương quan với biến mục tiêu.",
      "Xáo trộn ngẫu nhiên dữ liệu trước khi chia thành tập Huấn luyện và Kiểm tra.",
      "Sử dụng kỹ thuật K-fold cross-validation để tận dụng tối đa dữ liệu lịch sử.",
      "Sử dụng tập dữ liệu xác thực (validation) lớn hơn 20% tổng lượng dữ liệu."
    ],
    correctAnswer: 1,
    explanation: "Đối với dữ liệu chuỗi thời gian (time series), việc xáo trộn ngẫu nhiên sẽ dẫn đến 'temporal leak' (rò rỉ thời gian) - mô hình sẽ được huấn luyện trên dữ liệu tương lai để dự đoán quá khứ, một điều vô lý."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 15: Nếu mô hình của bạn hoàn toàn không thể giảm được Training Loss (mô hình không bắt đầu học được), nguyên nhân CÓ KHẢ NĂNG NHẤT là gì?",
    options: [
      "Tập dữ liệu huấn luyện của bạn đang bị thiếu các nhãn phân loại (labels).",
      "Vấn đề nằm ở cấu hình tốc độ học (learning rate) hoặc kích thước batch.",
      "Bạn đang sử dụng số lượng epoch huấn luyện lớn hơn mức cần thiết.",
      "Mô hình của bạn đang có kiến trúc quá sâu với quá nhiều tham số."
    ],
    correctAnswer: 1,
    explanation: "Mọi mô hình (kể cả với nhãn ngẫu nhiên) đều phải có khả năng giảm training loss. Nếu loss kẹt cứng ngay từ đầu, đó luôn là vấn đề của thuật toán gradient descent: Learning rate quá cao/quá thấp, hoặc Batch size không phù hợp."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 16: Một mô hình huấn luyện bình thường nhưng Validation Metrics không thể vượt qua 'Common-sense baseline'. Điều này ngụ ý gì?",
    options: [
      "Bạn cần tăng learning rate lên khoảng 10 lần để mô hình vượt ngưỡng.",
      "Mô hình của bạn có thể đang dùng sai giả định kiến trúc (architecture priors).",
      "Mô hình của bạn đã bị overfitting ngay từ vòng lặp epoch đầu tiên.",
      "Tập dữ liệu xác thực của bạn đang chứa quá nhiều mẫu trùng lặp."
    ],
    correctAnswer: 1,
    explanation: "Nếu mô hình không thắng nổi một baseline ngây thơ, nghĩa là nó hoàn toàn không nắm bắt được thông tin hữu ích. Nguyên nhân có thể do dữ liệu thực sự không chứa thông tin dự đoán, hoặc bạn đang dùng sai loại mô hình (vd: dùng Dense layer cho dữ liệu ảnh thay vì CNN)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 17: Để một mô hình máy học có thể khái quát hóa tốt, bước ĐẦU TIÊN bạn phải đạt được là gì?",
    options: [
      "Đảm bảo mô hình có thể Overfit (trang bị quá mức) trên tập huấn luyện.",
      "Triển khai tất cả các kỹ thuật chuẩn hóa (Regularization) có thể có.",
      "Thu giảm kích thước mạng nơ-ron xuống mức nhỏ nhất để chạy nhanh.",
      "Xóa bỏ hoàn toàn tập dữ liệu kiểm tra để dồn dữ liệu cho huấn luyện."
    ],
    correctAnswer: 0,
    explanation: "Quy tắc cốt lõi: Để tìm được điểm cân bằng hoàn hảo, trước tiên bạn phải vượt qua nó. Bạn bắt buộc phải thiết kế được một mô hình đủ lớn để có khả năng Overfit, sau đó mới dùng Regularization để kéo nó về mức tối ưu."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 18: Kỹ thuật Feature Engineering (Kỹ thuật Đặc trưng) mang lại lợi ích gì cho mô hình học sâu?",
    options: [
      "Nó làm tăng kích thước đa tạp ẩn, giúp mạng nơ-ron ghi nhớ nhiều thông tin hơn.",
      "Nó biến một bài toán đơn giản thành phức tạp để mô hình có thể khai thác sức mạnh.",
      "Nó giúp bài toán dễ học hơn, làm phẳng đa tạp ẩn và cần ít dữ liệu hơn.",
      "Nó tự động tinh chỉnh learning rate (tốc độ học) thay cho kỹ sư máy học."
    ],
    correctAnswer: 2,
    explanation: "Feature engineering là việc sử dụng tri thức con người để trích xuất đặc trưng có giá trị trước khi đưa vào mô hình (vd: tính tọa độ kim đồng hồ thay vì dùng ảnh gốc). Việc này giúp giảm gánh nặng học tập cho mạng nơ-ron và giảm lượng dữ liệu cần thiết."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 19: Early Stopping (Dừng sớm) là một kỹ thuật dùng để giải quyết vấn đề gì?",
    options: [
      "Tránh việc máy tính bị quá tải nhiệt khi huấn luyện mô hình quá lâu.",
      "Ngăn chặn mô hình rơi vào trạng thái Overfitting do huấn luyện quá nhiều epoch.",
      "Giải quyết tình trạng dữ liệu đầu vào bị thiếu hụt (missing values).",
      "Hạn chế việc mạng nơ-ron dự đoán sai các điểm dữ liệu mới."
    ],
    correctAnswer: 1,
    explanation: "Early stopping là kỹ thuật theo dõi Validation Loss. Nó sẽ tự động dừng quá trình huấn luyện ngay tại thời điểm epoch mà Validation Loss bắt đầu có dấu hiệu tăng lên (bắt đầu Overfitting)."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 20: Tác dụng cốt lõi của các kỹ thuật Regularization (Chuẩn hóa mô hình) là gì?",
    options: [
      "Làm cho mô hình có cấu trúc phức tạp hơn để học được nhiều chi tiết hơn.",
      "Ép mô hình trở nên đơn giản hơn, làm phẳng các đường cong quyết định.",
      "Chuyển đổi dữ liệu chuỗi thời gian thành dữ liệu bảng một cách tự động.",
      "Giúp thuật toán tăng tốc độ tính toán gradient lên mức tối đa có thể."
    ],
    correctAnswer: 1,
    explanation: "Regularization (giảm kích thước mạng, L1, L2, Dropout) hoạt động theo nguyên lý dao cạo Occam: nó phạt các cấu hình phức tạp, ép mô hình phải tìm kiếm các quy luật đơn giản hơn và có khả năng khái quát hóa tốt hơn."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 21: Cách ĐƠN GIẢN NHẤT để giảm thiểu Overfitting trong một mạng nơ-ron là gì?",
    options: [
      "Bổ sung thêm hàng loạt lớp Dropout với tỷ lệ 90% cho mọi layer.",
      "Giảm dung lượng mô hình bằng cách giảm số lớp hoặc số nơ-ron mỗi lớp.",
      "Thực hiện mã hóa One-hot cho toàn bộ các đặc trưng số thực.",
      "Tăng số lượng vòng lặp (epochs) lên gấp 10 lần so với ban đầu."
    ],
    correctAnswer: 1,
    explanation: "Việc thu nhỏ kích thước mạng sẽ làm giảm khả năng ghi nhớ máy móc (memorization capacity) của nó. Bị ép phải dùng ít tài nguyên, mô hình sẽ buộc phải tìm ra các đặc trưng cốt lõi thay vì nhớ vẹt."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 22: Chuẩn hóa trọng số L2 (L2 Regularization) trong Keras còn được gọi bằng thuật ngữ nào khác?",
    options: [
      "Weight penalty (Hình phạt trọng lượng)",
      "Dropout rate (Tỷ lệ loại bỏ)",
      "Weight decay (Suy giảm trọng số)",
      "Gradient clipping (Cắt xén gradient)"
    ],
    correctAnswer: 2,
    explanation: "L2 Regularization phạt mô hình tỷ lệ thuận với bình phương của các trọng số. Trong tối ưu hóa mạng nơ-ron, nó tương đương về mặt toán học với việc liên tục làm giảm trọng số, nên còn được gọi là Weight Decay."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 23: Lớp Dropout hoạt động dựa trên cơ chế nào TRONG QUÁ TRÌNH HUẤN LUYỆN (Training time)?",
    options: [
      "Thay thế toàn bộ trọng số của lớp đó bằng các giá trị ngẫu nhiên.",
      "Ngẫu nhiên thiết lập một tỷ lệ các đầu ra (activations) của lớp đó về giá trị 0.",
      "Tạm thời đóng băng không cập nhật gradient của lớp đó trong một vài epoch.",
      "Nhân tất cả các đầu ra của lớp đó với một hằng số tỷ lệ thuận."
    ],
    correctAnswer: 1,
    explanation: "Trong quá trình huấn luyện, Dropout ngẫu nhiên vô hiệu hóa (set về 0) một tỷ lệ nhất định các nơ-ron (vd: 50%). Việc này ngăn cản các nơ-ron tạo thành các 'âm mưu' cục bộ (đồng phụ thuộc lẫn nhau) để học vẹt."
  },
  {
    type: "fill_blank",
    difficulty: "Trung bình",
    question: "Câu 24: Khi sử dụng lớp Dropout với tỷ lệ 0.5 trong quá trình huấn luyện, vào thời điểm kiểm tra (Test time), mạng sẽ ________ đầu ra bằng một hệ số tương ứng (thường là nhân với 0.5) để cân bằng.",
    blanks: ["chia tỉ lệ", "scale down", "thu nhỏ", "nhân"],
    explanation: "Ở lúc Test, không có nơ-ron nào bị tắt (để có dự đoán tốt nhất). Nhưng vì số nơ-ron hoạt động nhiều gấp đôi lúc Train, ta phải 'scale down' giá trị đầu ra (nhân với 0.5) để giữ nguyên cường độ tín hiệu."
  },
  {
    type: "sorting",
    difficulty: "Khó",
    question: "Câu 25: Sắp xếp các bước chuẩn (Workflow) khi bạn xây dựng một mô hình học sâu từ đầu:",
    steps: [
      "Làm sạch dữ liệu, Feature Engineering và thiết lập Common-sense baseline.",
      "Xây dựng một mô hình đủ lớn và huấn luyện lâu để đạt trạng thái Overfit.",
      "Áp dụng Regularization (L2, Dropout) và giảm kích thước để cải thiện Khái quát hóa.",
      "Tinh chỉnh Early Stopping bằng Validation set và chốt mô hình cuối cùng."
    ],
    explanation: "Quy trình chuẩn: (1) Chuẩn bị dữ liệu và BaseLine -> (2) Phá vỡ rào cản Underfit để tiến tới Overfit -> (3) Gọt giũa bằng Regularization -> (4) Chốt số Epoch tối ưu bằng Early Stopping."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 26: Việc rò rỉ thông tin (Information leak) vào mô hình xảy ra do hành động nào?",
    options: [
      "Huấn luyện mô hình trực tiếp trên tập dữ liệu kiểm tra (Test set).",
      "Sử dụng kết quả Validation để điều chỉnh siêu tham số (Hyperparameters) nhiều lần.",
      "Dữ liệu huấn luyện vô tình bị sao chép hai lần trong hệ thống thư mục.",
      "Các trọng số của mô hình được lưu ra đĩa và tải lại vào bộ nhớ."
    ],
    correctAnswer: 1,
    explanation: "Mặc dù bạn không cho mô hình học trực tiếp Validation set, nhưng nếu bạn dùng điểm số Validation để sửa mô hình hàng chục lần, bạn đã vô tình cung cấp cấu trúc của tập Validation cho mô hình. Đó là Information leak."
  },
  {
    type: "mcq",
    difficulty: "Dễ",
    question: "Câu 27: Biện pháp hiệu quả NHẤT để giúp một mô hình học sâu khái quát hóa tốt là gì?",
    options: [
      "Bổ sung thuật toán Dropout với tỷ lệ loại bỏ lên mức 80%.",
      "Sử dụng thêm các lớp ẩn (Hidden layers) với hàng ngàn nơ-ron.",
      "Thu thập thêm dữ liệu huấn luyện mới và đảm bảo chất lượng nhãn.",
      "Chuyển từ thuật toán tối ưu hóa RMSprop sang thuật toán SGD."
    ],
    correctAnswer: 2,
    explanation: "Dữ liệu là vua. Học sâu chỉ là nội suy trên không gian dữ liệu. Việc lấy mẫu không gian đầu vào dày đặc hơn (thêm dữ liệu) luôn mang lại lợi ích khái quát hóa to lớn nhất so với mọi tinh chỉnh mô hình nào."
  },
  {
    type: "mcq",
    difficulty: "Trung bình",
    question: "Câu 28: Vì sao mạng nơ-ron sâu lại có khả năng học được các biểu diễn cực kỳ phức tạp?",
    options: [
      "Vì chúng là những hàm toán học gián đoạn, cho phép bẻ gãy không gian linh hoạt.",
      "Vì chúng được thiết kế để kết nối ngẫu nhiên các đặc trưng độc lập.",
      "Vì chúng tổ chức việc học một cách phân cấp và mô-đun hóa, tương tự cấu trúc tự nhiên.",
      "Vì mỗi nơ-ron hoạt động như một cỗ máy suy luận logic riêng biệt (Reasoning engine)."
    ],
    correctAnswer: 2,
    explanation: "Mạng nơ-ron học biểu diễn dữ liệu theo cách phân cấp (hierarchical) và mô-đun. Các lớp đầu học chi tiết nhỏ, các lớp sau học chi tiết phức tạp, phản ánh đúng cấu trúc phân tầng của thông tin trong thế giới thực."
  },
  {
    type: "mcq",
    difficulty: "Khó",
    question: "Câu 29: Nếu biểu đồ Loss của mô hình có Training Loss dao động mạnh nhưng Validation Loss lại đi ngang ở một mức cao không đổi, mô hình của bạn đang bị gì?",
    options: [
      "Mô hình đang học quá tốt và đã đạt điểm cực tiểu toàn cục.",
      "Mô hình bị thiếu tham số và không thể biểu diễn được dữ liệu.",
      "Mô hình đang bị Overfitting trầm trọng vào một tập dữ liệu cực nhỏ.",
      "Learning rate có thể đang được thiết lập ở mức quá cao."
    ],
    correctAnswer: 3,
    explanation: "Đồ thị Training Loss dao động mạnh (nhảy lên nhảy xuống không hội tụ) thường là dấu hiệu của việc Learning rate quá lớn, khiến quá trình Gradient Descent bị văng qua lại quanh điểm cực tiểu mà không thể tiến sâu vào trong."
  },
  {
    type: "matching",
    difficulty: "Trung bình",
    question: "Câu 30: Ghép nối giải pháp giải quyết cho từng trạng thái của mô hình:",
    pairs: [
      { left: "Training loss không giảm (Không học được)", right: "Tăng/giảm Learning rate hoặc tăng Batch size" },
      { left: "Validation loss không thắng được Baseline", right: "Sử dụng cấu trúc mạng (Architecture) phù hợp hơn" },
      { left: "Chỉ bị Underfit, không thể Overfit", right: "Tăng kích thước mạng (thêm lớp, thêm nơ-ron)" },
      { left: "Overfit quá nhanh và quá mạnh", right: "Giảm kích thước mạng, thêm L2, Dropout hoặc Early Stopping" }
    ],
    explanation: "Bảng tóm tắt các chiến lược Troubleshooting (sửa lỗi) kinh điển nhất trong kỹ thuật thiết kế và huấn luyện Học sâu."
  }
];

export default quizData;
