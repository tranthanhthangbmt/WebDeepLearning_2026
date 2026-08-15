const questions = [
    {
        type: "mcq",
        question: "Theo Chương 20, định nghĩa nào sau đây mô tả đúng nhất về 'Trí tuệ nhân tạo' (AI)?",
        options: [
            "Quá trình biến dữ liệu thành một chương trình một cách tự động thông qua các hàm mất mát.",
            "Tất cả những nỗ lực nhằm tự động hóa quá trình nhận thức của con người, từ bảng tính Excel đến robot.",
            "Một nhánh sử dụng chuỗi dài các phép biến đổi hình học có thể vi phân để phân loại dữ liệu.",
            "Khả năng tạo ra văn bản, hình ảnh, video và âm thanh từ các lời nhắc bằng ngôn ngữ tự nhiên."
        ],
        answer: 1,
        explanation: "AI là một lĩnh vực cực kỳ rộng lớn, bao gồm mọi nỗ lực tự động hóa nhận thức, bao gồm cả những cách tiếp cận dựa trên quy tắc đơn giản như hệ chuyên gia, chứ không chỉ riêng học máy hay học sâu.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Đâu KHÔNG phải là một trong những yếu tố chính thúc đẩy cuộc cách mạng của Học sâu theo tác giả?",
        options: [
            "Sự đổi mới dần dần về thuật toán qua hai thập kỷ, đỉnh cao là kiến trúc Transformer.",
            "Sự sẵn có của lượng lớn dữ liệu hình ảnh, video và văn bản nhờ sự phát triển của internet.",
            "Phát hiện ra thuật toán duy nhất có thể mô phỏng chính xác cấu trúc sinh học của bộ não.",
            "Sự phát triển của phần cứng tính toán song song giá rẻ (GPU) và các công cụ phần mềm như Keras."
        ],
        answer: 2,
        explanation: "Học sâu thực chất không mô phỏng bộ não con người. Tên gọi 'mạng nơ-ron' mang tính lịch sử. Sự thành công của nó đến từ phần cứng (GPU), lượng dữ liệu khổng lồ, công cụ phần mềm và cải tiến thuật toán.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Mô hình Generative AI (AI tạo sinh) học cách tái tạo dữ liệu thông qua cơ chế học tập ________ (tự giám sát), tức là tái tạo lại các phần bị thiếu của đầu vào.",
        answer: "tự giám sát",
        explanation: "AI tạo sinh (như LLM) sử dụng học tự giám sát (self-supervised learning) để dự đoán từ tiếp theo hoặc khử nhiễu hình ảnh mà không cần con người gắn nhãn thủ công.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Hình ảnh trực quan tốt nhất để mô tả quá trình biến đổi dữ liệu trong Học sâu là gì?",
        options: [
            "Quá trình gỡ một quả bóng giấy bị nhàu nát thành một tờ giấy phẳng.",
            "Một mạng lưới các đường ống nước có các van đóng mở tự động ngẫu nhiên.",
            "Quá trình tìm kiếm đường đi ngắn nhất giữa hai điểm trên một bản đồ đô thị.",
            "Một khu rừng nơi các thân cây mọc lên hoặc chết đi dựa trên nguồn ánh sáng."
        ],
        answer: 0,
        explanation: "Mỗi lớp học sâu thực hiện một biến đổi hình học nhỏ. Quá trình mạng nơ-ron phân tách các lớp dữ liệu phức tạp (đa tạp) giống như việc vuốt phẳng (uncrumpling) một quả bóng giấy bị nhàu.",
        difficulty: "Trung bình"
    },
    {
        type: "matching",
        question: "Ghép nối kiến trúc mạng (Network architecture) với loại dữ liệu phù hợp nhất để xử lý:",
        pairs: [
            { left: "Mạng kết nối dày đặc (Dense/MLP)", right: "Dữ liệu dạng bảng (Vector attributes)" },
            { left: "Mạng tích chập (ConvNet)", right: "Dữ liệu hình ảnh (Images)" },
            { left: "Mạng hồi quy (RNN/LSTM/GRU)", right: "Dữ liệu chuỗi thời gian liên tục" },
            { left: "Máy biến áp (Transformer)", right: "Văn bản và Dữ liệu đa phương thức (Sequence/Set)" }
        ],
        explanation: "Bảng 20.1: Mỗi kiến trúc mạng đại diện cho một không gian giả thuyết riêng biệt, tối ưu hóa cho cấu trúc không gian hoặc thời gian của một loại dữ liệu cụ thể.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Trong quy trình 7 bước xây dựng Học máy, sau khi 'Xác định vấn đề', bước quan trọng tiếp theo là gì?",
        options: [
            "Huấn luyện mô hình ngay lập tức để tạo ra một đường cơ sở ngẫu nhiên.",
            "Xác định một cách đo lường mức độ thành công (metric) một cách đáng tin cậy.",
            "Vector hóa dữ liệu thành các ma trận bằng cách sử dụng các hàm băm.",
            "Áp dụng chuẩn hóa hàng loạt vào mọi lớp mạng để ngăn chặn tình trạng overfitting."
        ],
        answer: 1,
        explanation: "Trước khi xây dựng mô hình, bạn phải biết đích đến của mình là gì và làm thế nào để đo lường nó (ví dụ: Độ chính xác, AUC, F1-score) dựa trên đặc thù của miền ứng dụng.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tại sao cần phải chia dữ liệu thành tập huấn luyện (training), tập xác thực (validation) và tập kiểm tra (test)?",
        options: [
            "Để rút ngắn thời gian tính toán của card đồ họa khi mô hình có kích thước quá lớn.",
            "Để tránh việc thiết lập siêu tham số bị overfitting (quá khớp) vào tập xác thực.",
            "Để cho phép sử dụng cùng lúc nhiều framework khác nhau như Keras, PyTorch và JAX.",
            "Để cân bằng lại số lượng mẫu giữa các lớp (class) bị thiểu số trong quá trình học."
        ],
        answer: 1,
        explanation: "Quá trình điều chỉnh các tham số dựa trên tập validation nhiều lần có thể làm mô hình 'học thuộc' tập validation đó (Information leak). Tập test độc lập được giữ lại cuối cùng để đánh giá thực tế.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Trong mạng kết nối dày đặc, để phân loại nhị phân (Binary classification), lớp Dense cuối cùng sử dụng hàm kích hoạt ________ và hàm mất mát binary_crossentropy.",
        answer: "sigmoid",
        explanation: "Hàm sigmoid nén đầu ra về khoảng [0, 1], biểu thị xác suất thuộc về phân lớp dương tính, hoàn toàn phù hợp với phân loại nhị phân.",
        difficulty: "Trung bình"
    },
    {
        type: "sorting",
        question: "Sắp xếp theo thứ tự quy trình xây dựng mô hình máy học phổ quát (Universal ML Workflow):",
        order: [
            "Xác định vấn đề, thu thập dữ liệu và chuẩn bị số liệu đánh giá (metrics).",
            "Phát triển quy trình đánh giá chéo (Validation process).",
            "Xây dựng mô hình đường cơ sở (Baseline model) để vượt qua dự đoán ngẫu nhiên.",
            "Mở rộng mô hình để đạt trạng thái quá khớp (Overfitting), sau đó sử dụng các kỹ thuật chính quy hóa."
        ],
        explanation: "Trước tiên phải có dữ liệu và mục tiêu. Sau đó tạo hệ thống đánh giá. Tiếp theo tạo mô hình nhỏ đủ để chứng minh thuật toán hoạt động. Cuối cùng tăng dung lượng mô hình rồi điều chỉnh lại bằng regularization.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Trong kiến trúc ConvNet, tại sao lại cần sử dụng các lớp tổng hợp (như MaxPooling2D)?",
        options: [
            "Để chuyển đổi dữ liệu từ dạng hình ảnh 2D sang dạng chuỗi 1D tuần tự.",
            "Để giảm độ phân giải không gian của bản đồ đặc trưng (downsample) và mở rộng trường nhìn.",
            "Để thêm các tín hiệu phi tuyến tính vào quá trình tích chập các giá trị pixel.",
            "Để chuẩn hóa phương sai của các lô dữ liệu trước khi đi vào mạng kết nối dày đặc."
        ],
        answer: 1,
        explanation: "Pooling giúp thu gọn kích thước không gian, giảm khối lượng tính toán và cho phép các bộ lọc (filter) ở các lớp sâu hơn 'nhìn' thấy các đặc điểm rộng hơn trong hình ảnh ban đầu.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Đặc điểm cốt lõi nào của Máy biến áp (Transformer) giúp nó xử lý ngôn ngữ tự nhiên tốt hơn RNN?",
        options: [
            "Nó sử dụng các bộ lọc tích chập 1D để phát hiện ngữ pháp từ trái qua phải.",
            "Nó sử dụng sự chú ý thần kinh (neural attention) để mô tả mọi từ bằng ngữ cảnh của các từ khác.",
            "Nó có khả năng tự động tạo ra thêm dữ liệu huấn luyện khi tập dữ liệu gốc không đủ lớn.",
            "Nó yêu cầu dữ liệu phải được chuẩn hóa về cùng một độ dài câu thông qua việc cắt bỏ các từ."
        ],
        answer: 1,
        explanation: "Transformer có thể xem xét đồng thời toàn bộ chuỗi (không phải tuần tự từng bước như RNN), sử dụng cơ chế Self-Attention để gán trọng số ý nghĩa cho mỗi từ dựa trên mọi từ khác trong câu.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Học sâu rất kém trong việc học các thuật toán mang tính lập trình, nó giải quyết bài toán bằng cách ________ (chứ không phải thực thi một logic cứng nhắc).",
        answer: "nội suy",
        explanation: "Học sâu thực chất là một cơ sở dữ liệu khổng lồ lưu trữ các đa tạp liên tục. Nó ánh xạ đầu vào thành đầu ra bằng nội suy hình học, không thể thực hiện các bước logic chặt chẽ (như carrying over trong phép cộng).",
        difficulty: "Trung bình"
    },
    {
        type: "matching",
        question: "Ghép nối định dạng Đầu vào -> Đầu ra với Kiến trúc mô hình tối ưu tương ứng:",
        pairs: [
            { left: "Văn bản -> Văn bản", right: "Máy biến áp (Transformer)" },
            { left: "Văn bản, Hình ảnh -> Hình ảnh", right: "Mô hình khuếch tán (Diffusion Model) hoặc VAE" },
            { left: "Hình ảnh -> Xác suất lớp", right: "Mạng tích chập (ConvNet)" },
            { left: "Chuỗi thời gian -> Hồi quy", right: "Mạng RNN (GRU/LSTM) hoặc Transformer" }
        ],
        explanation: "Sự phân chia này là cơ sở để thiết kế mạng. Transformer xuất sắc trong chuỗi (văn bản). ConvNet thống trị thị giác. Diffusion Models được dùng để tạo ảnh.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Điều nào mô tả ĐÚNG nhất lý do vì sao học sâu thường thất bại ở các nhiệm vụ ARC-AGI?",
        options: [
            "Vì các nhiệm vụ này yêu cầu mô hình phải điều khiển các robot trong không gian ba chiều thời gian thực.",
            "Vì mô hình không có khả năng thích nghi với sự mới lạ và chỉ có thể khái quát hóa cục bộ.",
            "Vì việc mô tả các lưới ma trận đòi hỏi mô hình ngôn ngữ phải có hơn 1000 tỷ tham số.",
            "Vì các bộ chuẩn ARC-AGI được thiết kế bằng ngôn ngữ lập trình mà Python không thể tương thích."
        ],
        answer: 1,
        explanation: "Học sâu rất dễ vỡ trước những thay đổi hời hợt hoặc tình huống hoàn toàn mới (ngoài phân phối huấn luyện) do nó thiếu khả năng trừu tượng lấy chương trình làm trung tâm (logic thực sự).",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Đề cập đến tương lai của AI, tác giả mô tả 'Mô hình kết hợp' (Hybrid models) là gì?",
        options: [
            "Là sự kết hợp giữa thuật toán lượng tử và học máy thống kê cổ điển.",
            "Sự tích hợp giữa mô-đun thuật toán (suy luận rời rạc) và mô-đun học sâu (trực giác không gian).",
            "Mô hình sử dụng cùng lúc mạng tích chập và mạng hồi quy trên cùng một lớp dữ liệu.",
            "Hệ thống cho phép cả con người và AI cùng chỉnh sửa chung một đoạn mã nguồn trực tiếp."
        ],
        answer: 1,
        explanation: "Tương lai của AI đòi hỏi sự giao thoa giữa hệ thống học sâu hiện tại (Hệ thống 1: trực giác, mô hình hình học) và các thuật toán logic (Hệ thống 2: suy luận rời rạc, tìm kiếm cấu trúc).",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tác giả so sánh tên gọi 'Mạng nơ-ron' (Neural network) với điều gì?",
        options: [
            "Nó là một mô hình sinh học tuyệt đối chính xác của bộ não con người.",
            "Nó là một cái tên gây hiểu lầm sâu sắc vì nó hầu như không liên quan đến bộ não.",
            "Nó là công nghệ độc quyền được phát minh bởi các nhà khoa học thần kinh vào thế kỷ 20.",
            "Nó chỉ là một thuật ngữ marketing được sử dụng riêng biệt cho lĩnh vực xe tự lái."
        ],
        answer: 1,
        explanation: "Tác giả nhấn mạnh 'Mạng nơ-ron' là một cái tên lịch sử gây hiểu lầm. Nên gọi nó là 'Học biểu diễn phân tầng' hoặc 'Biến đổi hình học chuỗi' vì cốt lõi của nó là toán học (đại số tuyến tính) chứ không phải sinh học.",
        difficulty: "Trung bình"
    },
    {
        type: "sorting",
        question: "Sắp xếp quá trình hoạt động của Máy biến áp (Transformer) dạng tuần tự (Seq2Seq):",
        order: [
            "Nhúng chuỗi nguồn và thêm mã hóa vị trí (Positional Encoding).",
            "TransformerEncoder biến đổi chuỗi nguồn thành các vectơ nhận biết ngữ cảnh.",
            "TransformerDecoder nhận chuỗi đích (tính đến hiện tại) và đầu ra của Encoder.",
            "Lớp Dense dự đoán từ tiếp theo trong chuỗi đích (Softmax)."
        ],
        explanation: "Đây là kiến trúc Encoder-Decoder tiêu chuẩn của Transformer dùng trong dịch máy hoặc tạo văn bản.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Để theo dõi nghiên cứu mới nhất, tác giả khuyên nên đọc các bài báo truy cập mở trên máy chủ in sẵn có tên là ________.",
        answer: "arXiv",
        explanation: "arXiv là kho lưu trữ trung tâm của ngành khoa học máy tính và AI, nơi các bài báo được công bố công khai trước khi peer-review, giúp lĩnh vực này phát triển thần tốc.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tại sao nền tảng Kaggle lại được khuyến nghị trong phần kết luận?",
        options: [
            "Vì đây là nơi cung cấp các khóa học lý thuyết về toán cao cấp miễn phí.",
            "Vì nó cung cấp kinh nghiệm thực tế qua các cuộc thi giải quyết vấn đề bằng code thật.",
            "Vì Kaggle là công cụ độc quyền để chạy các thư viện Keras và TensorFlow cục bộ.",
            "Vì người dùng bắt buộc phải dùng Kaggle để được cấp chứng chỉ lập trình viên AI."
        ],
        answer: 1,
        explanation: "Kaggle cung cấp dữ liệu thực tế và tính cạnh tranh, buộc bạn phải vận dụng mọi kỹ thuật (tuning siêu tham số, chống overfitting, ensemble) để chiến thắng, đây là cách thực hành tốt nhất.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tác giả nhận định thế nào về câu nói 'Sự phát triển của AGI (Trí tuệ nhân tạo tổng quát) sẽ dẫn đến ngày tận thế robot độc nhất'?",
        options: [
            "Đó là một mối lo ngại thực sự và chúng ta phải dừng phát triển AI ngay lập tức.",
            "Đó là sự thật đã được các nhà khoa học chứng minh bằng các phép tính toán học phức tạp.",
            "Đó chỉ là ảo tưởng thuần túy xuất phát từ sự hiểu lầm sâu sắc về cả trí tuệ và công nghệ.",
            "Đó là điều chắc chắn xảy ra nếu chúng ta kết nối AI trực tiếp với hệ thống internet."
        ],
        answer: 2,
        explanation: "Theo tác giả (François Chollet), nỗi sợ về AGI thống trị thế giới là sự nhân cách hóa AI một cách thái quá. AI chỉ là công cụ tính toán và việc chúng trở nên ác ý giống như một bộ phim viễn tưởng thiếu cơ sở khoa học.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Khái niệm 'Sự trừu tượng lấy chương trình làm trung tâm' (Program-centric abstraction) khác biệt như thế nào so với học sâu hiện tại?",
        options: [
            "Nó đòi hỏi các cấu trúc phần cứng bằng silicon mô phỏng chính xác não người.",
            "Nó có khả năng vận hành thông qua luồng điều khiển, vòng lặp đệ quy và cấu trúc dữ liệu.",
            "Nó chỉ hoạt động với dữ liệu liên tục thay vì dữ liệu rời rạc phân tán.",
            "Nó phải được thực thi hoàn toàn bằng con người thay vì máy tính tự động."
        ],
        answer: 1,
        explanation: "Program-centric có cấu trúc như mã phần mềm (rời rạc, logic, biến số, vòng lặp). Học sâu (Value-centric) dựa trên khoảng cách hình học liên tục và không có các cấu trúc lý luận cứng nhắc này.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Theo tác giả, thành công của Deep Learning có thể được so sánh giống với cuộc cách mạng nào trong lịch sử?",
        options: [
            "Cuộc cách mạng công nghiệp lần thứ nhất ở Anh.",
            "Sự ra đời của Internet, bị cường điệu ngắn hạn nhưng sẽ biến đổi hoàn toàn nền kinh tế dài hạn.",
            "Sự phát minh ra máy bay và ngành hàng không vũ trụ hiện đại.",
            "Việc phát hiện ra Penicillin và công nghệ sinh học di truyền."
        ],
        answer: 1,
        explanation: "Tác giả cho rằng hiện tượng 'mùa hè AI' này rất giống với bong bóng Dot-com: có thể sẽ có sự điều chỉnh do cường điệu quá mức, nhưng công nghệ lõi sẽ tái định hình vĩnh viễn mọi thứ xung quanh chúng ta.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Trong quy trình ML, thay vì dự đoán nhãn cho mục tiêu phân loại có nhiều lớp, chúng ta sẽ dùng hàm mất mát categorical_crossentropy và hàm kích hoạt ________ ở lớp Dense cuối cùng.",
        answer: "softmax",
        explanation: "Softmax biến một vectơ các điểm số thành phân phối xác suất (tổng bằng 1), rất thích hợp cho bài toán phân loại đa lớp nhãn đơn.",
        difficulty: "Trung bình"
    },
    {
        type: "matching",
        question: "Ghép nối định nghĩa của các khái niệm toán học cốt lõi trong Học sâu:",
        pairs: [
            { left: "Đa tạp (Manifold)", right: "Bề mặt hình học liên tục, nhăn nheo của dữ liệu không gian nhiều chiều." },
            { left: "Hàm khoảng cách (Distance function)", right: "Công cụ toán học dùng để định lượng 'ý nghĩa' bằng cách đo đạc sự tương đồng cặp đôi." },
            { left: "Có thể vi phân (Differentiable)", right: "Tính chất liên tục và mượt mà, cho phép sử dụng đạo hàm để tối ưu hóa mô hình." },
            { left: "Giảm độ dốc (Gradient descent)", right: "Thuật toán cập nhật trọng số từng bước để tìm cực tiểu của hàm mất mát." }
        ],
        explanation: "Toàn bộ việc học sâu là quá trình 'gỡ rối' (uncrumpling) đa tạp dữ liệu thông qua các phép biến đổi hình học có thể vi phân, được dẫn dắt bởi hàm mất mát và giảm độ dốc.",
        difficulty: "Trung bình"
    },
    {
        type: "sorting",
        question: "Quy trình dự đoán của tương lai 'Tái kết hợp mô-đun và học tập suốt đời':",
        order: [
            "Hệ thống bắt gặp một nhiệm vụ hoàn toàn mới mẻ trong thế giới thực.",
            "Hệ thống tìm nạp các mô-đun hình học và thuật toán từ một thư viện khổng lồ toàn cầu.",
            "Hệ thống tự động kết hợp lại (recombine) chúng trên không gian chương trình để tạo ra giải pháp.",
            "Lưu lại các thành phần mới đã khám phá được vào thư viện cho những lần dùng sau."
        ],
        explanation: "Đây là giải pháp tiềm năng cho AGI: thay vì đào tạo một mạng khổng lồ từ con số 0, nó sẽ tiến hóa bằng cách lắp ráp các thư viện chương trình con.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tại sao phương pháp mở rộng quy mô (Scaling models and data) lại không đủ để tạo ra AGI?",
        options: [
            "Vì năng lượng trên Trái đất không đủ cung cấp cho mạng điện toán lớn đến mức đó.",
            "Vì mô hình chỉ dùng nội suy tĩnh, nó sẽ tăng bộ nhớ cục bộ nhưng vẫn thất bại hoàn toàn trước tình huống mới lạ.",
            "Vì dữ liệu trên Internet chứa quá nhiều thông tin độc hại không thể lọc bỏ hoàn toàn.",
            "Vì các kỹ sư hiện tại chưa tìm ra ngôn ngữ lập trình đủ tốt thay thế cho Python."
        ],
        answer: 1,
        explanation: "Quy mô khổng lồ chỉ giúp AI học thuộc lòng và làm tốt khi gặp các dạng bài tương tự (khái quát hóa cục bộ). Nó không tạo ra khả năng lý luận linh hoạt (trí thông minh) cho các ẩn số chưa biết (khái quát hóa cực độ).",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Trong tương lai, deep learning sẽ đóng vai trò gì đối với cộng đồng công nghệ phần mềm nói chung?",
        options: [
            "Chỉ các nhà nghiên cứu có bằng Tiến sĩ mới được phép sử dụng công nghệ này.",
            "Nó sẽ được trừu tượng hóa và trở thành công cụ tiêu chuẩn trong hộp công cụ của mọi lập trình viên giống như công nghệ web ngày nay.",
            "Nó sẽ tự động viết mọi mã nguồn và lập trình viên con người sẽ hoàn toàn biến mất.",
            "Nó sẽ bị loại bỏ dần và thay thế bởi các thuật toán mã hóa cứng (hard-coded algorithms)."
        ],
        answer: 1,
        explanation: "Các công cụ như Keras đóng vai trò quan trọng trong việc dân chủ hóa AI. Bất kỳ lập trình viên nào cũng có thể nhúng các thành phần nhận thức trí tuệ vào ứng dụng của họ mà không cần là chuyên gia toán học.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Sự phân loại đa nhãn (Multi-label classification), ví dụ như gắn nhiều tag cho một hình ảnh, sử dụng hàm mất mát nào?",
        options: [
            "categorical_crossentropy",
            "binary_crossentropy",
            "mean_squared_error",
            "sparse_categorical_crossentropy"
        ],
        answer: 1,
        explanation: "Trong Multi-label, mỗi lớp (nhãn) hoạt động độc lập như một bài toán phân loại nhị phân. Do đó, ta phải dùng sigmoid kích hoạt và binary_crossentropy.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Để đo lường hiệu quả mô hình, bước phát triển một mô hình đầu tiên để chứng minh rằng Học máy có thể giải quyết được vấn đề được gọi là xây dựng đường ________ (baseline).",
        answer: "cơ sở",
        explanation: "Trước khi xây dựng các mạng deep learning phức tạp, bạn phải có một mô hình cơ sở (baseline) sử dụng các luật đơn giản hoặc máy học cơ bản (như hồi quy tuyến tính) để xác định xem bài toán có khả thi hay không.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Nếu tác giả của Keras (François Chollet) đúng, trí tuệ nhân tạo tương lai (AGI) sẽ KHÔNG phụ thuộc vào điều gì?",
        options: [
            "Sự phát triển của phương pháp tổng hợp chương trình (Program Synthesis).",
            "Việc sử dụng trực giác từ học sâu để hướng dẫn tìm kiếm trong không gian lớn.",
            "Chỉ việc mở rộng sức mạnh tính toán Brute-force trên dữ liệu tĩnh mà không cần sáng tạo cấu trúc.",
            "Quá trình học tập suốt đời (Lifelong learning) và tích lũy các mô-đun."
        ],
        answer: 2,
        explanation: "Chollet cực lực phản đối ý tưởng 'Scale is all you need' (Chỉ cần quy mô là đủ). Trí thông minh là tính hiệu quả, chứ không phải dùng sức mạnh máy tính thô bạo (Brute-force computing) để đoán đáp án.",
        difficulty: "Trung bình"
    }
];

export default questions;

