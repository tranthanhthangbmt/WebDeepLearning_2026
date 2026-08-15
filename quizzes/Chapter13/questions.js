const questions = [
    {
        question: "Đâu là nhiệm vụ phổ biến nhất liên quan đến dữ liệu chuỗi thời gian?",
        options: [
            "Dự báo tương lai (Forecasting)",
            "Phân loại (Classification)",
            "Phát hiện bất thường (Anomaly detection)",
            "Phát hiện sự kiện (Event detection)"
        ],
        answer: 0,
        explanation: "Dự báo (dự đoán điều gì xảy ra tiếp theo trong chuỗi) là nhiệm vụ phổ biến nhất liên quan đến chuỗi thời gian.",
        difficulty: "Dễ",
        type: "mcq"
    },
    {
        question: "Phát hiện hoạt động bất thường trên mạng công ty hoặc thông số bất thường trên dây chuyền sản xuất thuộc loại nhiệm vụ nào của chuỗi thời gian?",
        options: [
            "Phát hiện bất thường (Anomaly detection)",
            "Dự báo tương lai (Forecasting)",
            "Phân loại (Classification)",
            "Phát hiện sự kiện (Event detection)"
        ],
        answer: 0,
        explanation: "Phát hiện bất thường thường được thực hiện thông qua học tập không giám sát để phát hiện các sự kiện bất thường hoặc không bình thường.",
        difficulty: "Dễ",
        type: "mcq"
    },
    {
        question: "Ứng dụng 'phát hiện từ nóng' như phát hiện câu nói 'OK, Google' thuộc loại nhiệm vụ nào của chuỗi thời gian?",
        options: [
            "Phát hiện sự kiện (Event detection)",
            "Phân loại (Classification)",
            "Phát hiện bất thường (Anomaly detection)",
            "Dự báo tương lai (Forecasting)"
        ],
        answer: 0,
        explanation: "Phát hiện sự kiện xác định sự xuất hiện của một sự kiện cụ thể được mong đợi trong một khoảng thời gian liên tục.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Trong tập dữ liệu thời tiết Jena, dữ liệu được ghi lại với tần suất như thế nào?",
        options: [
            "10 phút một lần",
            "1 giờ một lần",
            "1 ngày một lần",
            "1 phút một lần"
        ],
        answer: 0,
        explanation: "Bộ dữ liệu thời tiết Jena bao gồm 14 đại lượng khác nhau được ghi lại 10 phút một lần trong vài năm.",
        difficulty: "Dễ",
        type: "mcq"
    },
    {
        question: "Đặc tính quan trọng và rất phổ biến nào của dữ liệu chuỗi thời gian thường thấy ở dữ liệu thời tiết hoặc mức tiêu thụ điện?",
        options: [
            "Tính tuần hoàn trên nhiều khoảng thời gian khác nhau",
            "Tính ngẫu nhiên hoàn toàn không có bất kỳ quy luật nào",
            "Sự thay đổi theo xu hướng tuyến tính tăng dần liên tục",
            "Phân phối chuẩn đồng nhất tại mọi thời điểm đo lường"
        ],
        answer: 0,
        explanation: "Tính tuần hoàn trên nhiều khoảng thời gian (như chu kỳ hàng ngày và hàng năm) là một đặc tính quan trọng của chuỗi thời gian, giúp tạo ra tính định kỳ có thể dự đoán.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Khi chia dữ liệu chuỗi thời gian thành các tập huấn luyện, xác thực và kiểm tra, điều kiện quan trọng nào cần được tuân thủ?",
        options: [
            "Dữ liệu xác thực và kiểm tra phải mới hơn dữ liệu huấn luyện",
            "Các tập dữ liệu phải được phân chia hoàn toàn ngẫu nhiên",
            "Dữ liệu huấn luyện phải xen kẽ đều với dữ liệu kiểm tra",
            "Kích thước của tập dữ liệu huấn luyện phải bằng tập kiểm tra"
        ],
        answer: 0,
        explanation: "Khi làm việc với chuỗi thời gian, điều quan trọng là dữ liệu xác thực và kiểm tra mới hơn dữ liệu huấn luyện vì chúng ta đang dự đoán tương lai dựa trên quá khứ.",
        difficulty: "Dễ",
        type: "mcq"
    },
    {
        question: "Khi chuẩn hóa dữ liệu chuỗi thời gian, chúng ta tính giá trị trung bình và độ lệch chuẩn trên tập dữ liệu nào?",
        options: [
            "Chỉ trên tập dữ liệu huấn luyện gốc",
            "Trên toàn bộ ba tập dữ liệu phân chia",
            "Chỉ trên tập dữ liệu kiểm tra cuối",
            "Trên tập dữ liệu xác thực trung gian"
        ],
        answer: 0,
        explanation: "Chúng ta chỉ tính giá trị trung bình và độ lệch chuẩn trên phần dữ liệu huấn luyện để tránh làm rò rỉ thông tin tương lai vào mô hình.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Tại sao nên thử một phương pháp 'thông thường' (đường cơ sở thông thường) trước khi sử dụng các mô hình học máy phức tạp?",
        options: [
            "Để thiết lập một đường cơ sở làm tiêu chuẩn chứng minh tính hữu ích",
            "Để ngay lập tức giải quyết vấn đề bằng độ chính xác cao tuyệt đối",
            "Vì các mô hình học máy luôn luôn có kết quả kém hơn đường cơ sở",
            "Để giảm thời gian thiết kế mạng của các mô hình Deep Learning sau này"
        ],
        answer: 0,
        explanation: "Đường cơ sở thông thường đóng vai trò như một cuộc kiểm tra độ tỉnh táo và thiết lập mức tối thiểu mà các mô hình tiên tiến hơn cần phải vượt qua.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Trong bài toán dự báo nhiệt độ của chương 13, phương pháp cơ sở thông thường (naive baseline) là gì?",
        options: [
            "Dự đoán nhiệt độ 24 giờ tới sẽ giống hệt như nhiệt độ hiện tại",
            "Dự đoán nhiệt độ bằng trung bình cộng của toàn bộ dữ liệu mẫu",
            "Dự đoán nhiệt độ sẽ thay đổi theo đường tuyến tính tăng dần",
            "Sử dụng mạng thần kinh để đưa ra một giá trị nhiệt độ cố định"
        ],
        answer: 0,
        explanation: "Dựa trên tính liên tục của nhiệt độ và chu kỳ hàng ngày, phương pháp cơ sở thông thường giả định nhiệt độ 24 giờ tới giống hệt nhiệt độ hiện tại.",
        difficulty: "Dễ",
        type: "mcq"
    },
    {
        question: "Lý do chính khiến mạng được kết nối dày đặc (Dense) xử lý kém đối với dữ liệu chuỗi thời gian là gì?",
        options: [
            "Chúng làm phẳng các chuỗi và loại bỏ khái niệm thứ tự thời gian",
            "Chúng có quá ít tham số cấu trúc để học các chu kỳ thời gian",
            "Chúng chỉ hoạt động hiệu quả khi đối phó với dữ liệu hình ảnh",
            "Chúng yêu cầu kích thước lô dữ liệu phải lớn vô hạn để huấn luyện"
        ],
        answer: 0,
        explanation: "Mạng kết nối dày đặc trước tiên làm phẳng (flatten) các chuỗi thời gian thành một vector duy nhất, qua đó loại bỏ khái niệm thời gian khỏi đầu vào.",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Lý do chính khiến mạng tích chập 1D (1D ConvNet) hoạt động không tốt trên dữ liệu dự báo nhiệt độ Jena là gì?",
        options: [
            "Dữ liệu không bất biến dịch chuyển và pooling phá hủy thông tin đơn hàng",
            "Mạng 1D ConvNet chỉ có thể sử dụng lớp chập mà không có lớp Pooling",
            "Mạng tích chập không thể xử lý đầu vào có độ dài vượt quá giới hạn",
            "ConvNet 1D quá chậm trên CPU khi so sánh trực tiếp với các mạng Dense"
        ],
        answer: 0,
        explanation: "Mạng tích chập gặp vấn đề vì dữ liệu thời tiết chỉ bất biến dịch trong thời gian ngắn (ban ngày khác ban đêm), và các lớp Pooling phá hủy thông tin thứ tự thời gian, trong khi đối với chuỗi thời gian, thứ tự rất quan trọng.",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Đặc điểm khác biệt cốt lõi của mạng thần kinh tái phát (RNN) so với mạng tiếp liệu (feedforward) là gì?",
        options: [
            "Nó có vòng lặp bên trong để xử lý chuỗi tăng dần và duy trì trạng thái",
            "Nó làm phẳng toàn bộ đầu vào trước khi đưa qua lớp Dense duy nhất",
            "Nó xử lý toàn bộ đoạn chuỗi cùng một lúc mà không chia theo thời gian",
            "Nó chỉ có thể được huấn luyện thành công thông qua hàm mất mát tối ưu"
        ],
        answer: 0,
        explanation: "RNN xử lý các chuỗi bằng cách lặp qua các phần tử và duy trì trạng thái chứa thông tin liên quan đến những gì nó đã thấy, khác với feedforward xử lý mọi đầu vào độc lập.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Trong mã giả triển khai RNN, đầu ra của vòng lặp ở bước thời gian t được tính bằng cách nào?",
        options: [
            "Kết hợp đầu vào hiện tại với trạng thái hiện tại bằng một hàm kích hoạt",
            "Kết hợp trung bình của tất cả các đầu vào trước với đầu vào hiện tại",
            "Bỏ qua trạng thái hiện tại và chỉ tập trung sử dụng đầu vào tại bước t",
            "Tính tích vô hướng của đầu vào hiện tại với một đầu vào ngẫu nhiên"
        ],
        answer: 0,
        explanation: "Trong RNN, output_t = activation(dot(W, input_t) + dot(U, state_t) + b). Nó kết hợp đầu vào hiện tại (input_t) và trạng thái hiện tại (state_t).",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Vấn đề lớn nhất của lớp SimpleRNN khi xử lý các chuỗi dài trong thực tế là gì?",
        options: [
            "Gặp phải vấn đề lớn về độ dốc biến mất do các tín hiệu cũ giảm dần",
            "Không thể triển khai tính toán tăng tốc trên GPU do vòng lặp nội bộ",
            "Yêu cầu dung lượng bộ nhớ lớn để lưu trữ trạng thái của tất cả các bước",
            "Nó chỉ hỗ trợ thiết lập đầu ra với số chiều tương đương chiều đầu vào"
        ],
        answer: 0,
        explanation: "Giống như mạng nơ-ron truyền thẳng có quá nhiều lớp, SimpleRNN bị hiện tượng vanishing gradients, khiến nó không thể học được các sự phụ thuộc lâu dài.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Mục đích chính của luồng dữ liệu mang (carry track) trong cấu trúc Bộ nhớ ngắn hạn dài (LSTM) là gì?",
        options: [
            "Lưu trữ thông tin an toàn để ngăn chặn các tín hiệu cũ biến mất hoàn toàn",
            "Tính toán độ cong của đồ thị mất mát nhằm phục vụ cho thao tác dừng sớm",
            "Làm phẳng đầu vào không gian hai chiều thành dạng chuỗi để mạng dễ học",
            "Xóa bỏ các điểm dữ liệu dị thường hoặc những nhiễu ngẫu nhiên trong chuỗi"
        ],
        answer: 0,
        explanation: "Luồng carry (Ct) cho phép thông tin từ quá khứ nhảy lên và di chuyển tới các bước thời gian sau đó một cách nguyên vẹn, giúp giải quyết vấn đề độ dốc biến mất.",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Lớp lặp lại nào trong Keras được coi là một phiên bản đơn giản và hợp lý hơn một chút so với kiến trúc LSTM?",
        options: [
            "Lớp GRU (Gated Recurrent Unit)",
            "Lớp SimpleRNN cơ bản",
            "Lớp Bidirectional nâng cao",
            "Lớp Conv1D truyền thống"
        ],
        answer: 0,
        explanation: "Đơn vị tái phát có cổng (GRU) rất giống với LSTM nhưng là một phiên bản đơn giản hơn, hợp lý hơn.",
        difficulty: "Dễ",
        type: "mcq"
    },
    {
        question: "Theo nghiên cứu của Yarin Gal, cách áp dụng Dropout đúng đắn cho mạng thần kinh tái phát (RNN) là gì?",
        options: [
            "Áp dụng cùng một mặt nạ dropout y hệt nhau ở tất cả mọi bước thời gian",
            "Sử dụng một mặt nạ dropout thay đổi hoàn toàn ngẫu nhiên ở mỗi bước mới",
            "Chỉ áp dụng dropout vào lớp cuối cùng và cấm dùng trong lớp lặp lại",
            "Sử dụng dropout riêng biệt biệt lập cho từng đoạn chuỗi ngắn trong lô"
        ],
        answer: 0,
        explanation: "Việc sử dụng cùng một mặt nạ bỏ học ở mọi bước thời gian cho phép mạng truyền bá lỗi học một cách chính xác theo thời gian, thay vì làm gián đoạn tín hiệu lỗi.",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Đối số nào trong lớp lặp lại của Keras được sử dụng để chỉ định tỷ lệ bỏ học (dropout) của các đơn vị tái diễn (recurrent units)?",
        options: [
            "recurrent_dropout",
            "dropout",
            "recurrent_rate",
            "state_dropout"
        ],
        answer: 0,
        explanation: "Trong Keras, 'dropout' kiểm soát tỷ lệ bỏ học cho các đầu vào của lớp, còn 'recurrent_dropout' kiểm soát bỏ học áp dụng cho các đơn vị tái diễn bên trong.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Khi xếp chồng nhiều lớp RNN lên nhau trong Keras, bạn phải thiết lập đối số nào của các lớp trung gian để chúng hoạt động?",
        options: [
            "return_sequences=True",
            "return_state=True",
            "sequence_output=True",
            "unroll=True"
        ],
        answer: 0,
        explanation: "Các lớp RNN trung gian cần truyền toàn bộ chuỗi cho lớp RNN tiếp theo, do đó cần phải kích hoạt return_sequences=True để trả về tenxơ cấp 3.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Nhược điểm lớn nhất của việc dùng tham số recurrent_dropout trong các lớp LSTM hoặc GRU trên thiết bị GPU là gì?",
        options: [
            "Nó không được hạt nhân cuDNN hỗ trợ nhanh chóng và buộc dùng TensorFlow",
            "Nó làm tăng kích thước tham số của mô hình lên gấp đôi trong mọi trường hợp",
            "Nó yêu cầu dữ liệu chuỗi thời gian phải được chuẩn hóa rất kỹ lưỡng từ đầu",
            "Nó khiến mạng không thể áp dụng cho các chuỗi thời gian có chu kỳ liên tục"
        ],
        answer: 0,
        explanation: "Hạt nhân cuDNN rất nhanh nhưng không hỗ trợ recurrent_dropout, khiến quá trình tính toán phải quay lại triển khai TensorFlow thông thường và chậm hơn 2-5 lần.",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Trong Keras, để tăng tốc lớp RNN khi không thể sử dụng cuDNN trên các chuỗi nhỏ, ta có thể sử dụng giải pháp nào?",
        options: [
            "Hủy cuộn vòng lặp bằng cách truyền tham số unroll=True vào khai báo lớp RNN",
            "Sử dụng thêm một lớp MaxPooling1D để làm ngắn chuỗi trước khi đi vào mạng",
            "Sử dụng thuật toán tối ưu Adam hiện đại thay vì thuật toán RMSprop cũ hơn",
            "Tăng kích thước batch size lên mức tối đa để tận dụng sức mạnh xử lý song song"
        ],
        answer: 0,
        explanation: "Cài đặt unroll=True sẽ tháo cuộn vòng lặp for của RNN, giúp TensorFlow tối ưu biểu đồ tính toán với điều kiện số lượng bước thời gian là cố định.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Loại dữ liệu nào thì RNN hai chiều (Bidirectional) thường mang lại lợi ích lớn và vượt trội hơn so với RNN theo thứ tự thời gian?",
        options: [
            "Dữ liệu ngôn ngữ văn bản nơi chiều đọc trước-sau không làm mất đi ý nghĩa",
            "Dữ liệu thời tiết lấy mẫu theo thời gian thực tại một địa phương nhất định",
            "Dữ liệu dự báo mức tiêu thụ điện năng hằng giờ của toàn bộ lưới điện",
            "Dữ liệu hình ảnh hai chiều tĩnh được thu thập bởi các thiết bị cảm biến"
        ],
        answer: 0,
        explanation: "Với văn bản, đọc ngược hay xuôi không phá hủy ý nghĩa tổng thể. RNN hai chiều nhìn câu từ hai hướng, nắm bắt nhiều biểu diễn hơn, trong khi dữ liệu thời tiết bị giảm hiệu năng nếu đọc ngược.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Tại sao mạng RNN hai chiều lại có hiệu năng khá kém trong bài toán dự báo nhiệt độ của bộ dữ liệu thời tiết Jena?",
        options: [
            "Một nửa của mạng đọc theo chiều đảo ngược thời gian không đem lại lợi ích",
            "Dữ liệu nhiệt độ không có tính chu kỳ nên mạng hai chiều không nhận diện được",
            "Lớp Bidirectional trong Keras không hỗ trợ xử lý chuỗi có độ dài lớn hơn mức trần",
            "Lớp Bidirectional bắt buộc phải đi kèm với mạng tích chập 1D để trích xuất trước"
        ],
        answer: 0,
        explanation: "Quá khứ gần đây rất quan trọng với dự báo thời tiết. Đọc ngược thời gian vô tác dụng trên dữ liệu này, đồng thời việc gấp đôi lượng tham số gây hiện tượng overfitting sớm.",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Theo chương 13, tại sao việc sử dụng học máy để cố gắng dự đoán giá chứng khoán thường là một ngõ cụt?",
        options: [
            "Trong thị trường này hiệu suất quá khứ không phải là cơ sở dự đoán cho tương lai",
            "Dữ liệu chứng khoán có quá ít điểm quan sát để huấn luyện đủ sâu một mạng LSTM",
            "Dữ liệu tài chính đặc thù không thể được chuẩn hóa như dữ liệu đo lường thời tiết",
            "Không có hệ thống mạng nơ-ron nào đủ sức phân tích độ phức tạp của chứng khoán"
        ],
        answer: 0,
        explanation: "Máy học rất tốt với dữ liệu mà quá khứ định hình tương lai. Tuy nhiên thị trường chứng khoán dựa trên thông tin chênh lệch, giá quá khứ không giúp dự đoán được tỷ suất sinh lời tương lai.",
        difficulty: "Trung bình",
        type: "mcq"
    },
    {
        question: "Hàm nào trong Keras hỗ trợ tự động sinh tập dữ liệu (dataset) với các cửa sổ được trích xuất từ mảng dữ liệu chuỗi thời gian ban đầu?",
        options: [
            "keras.utils.timeseries_dataset_from_array",
            "keras.preprocessing.sequence.pad_sequences",
            "keras.layers.TimeDistributed",
            "keras.utils.image_dataset_from_directory"
        ],
        answer: 0,
        explanation: "Hàm timeseries_dataset_from_array tự động sinh ra các chuỗi cửa sổ, giúp tiết kiệm bộ nhớ thay vì cấp phát rõ ràng tất cả các mẫu.",
        difficulty: "Dễ",
        type: "mcq"
    },
    {
        question: "Trong cấu trúc bộ nhớ của LSTM, có bao nhiêu phép biến đổi có dạng ô SimpleRNN được sử dụng để điều chỉnh thông tin của luồng mang?",
        options: [
            "Ba phép biến đổi với các ma trận trọng số i, f, và k",
            "Một phép biến đổi duy nhất để tối ưu hóa việc truyền tin",
            "Hai phép biến đổi cho đầu vào mới và cho trạng thái cũ",
            "Bốn phép biến đổi riêng biệt cho bốn cổng bộ nhớ song song"
        ],
        answer: 0,
        explanation: "LSTM tính toán giá trị tiếp theo của luồng dữ liệu mang bằng cách kết hợp ba phép biến đổi riêng biệt (i_t, f_t, k_t).",
        difficulty: "Khó",
        type: "mcq"
    },
    {
        question: "Ghép nối các loại lớp mạng với đặc điểm xử lý đặc trưng của chúng trên dữ liệu chuỗi thời gian:",
        options: [
            "Mạng kết nối dày đặc (Dense)",
            "Mạng tích chập 1D (1D ConvNet)",
            "Mạng lặp lại cơ bản (SimpleRNN)",
            "Mạng bộ nhớ ngắn hạn dài (LSTM)"
        ],
        answer: [
            "Làm phẳng chuỗi và loại bỏ khái niệm chiều thời gian của đầu vào",
            "Trượt qua dữ liệu tuần tự nhưng phá hủy thông tin đơn hàng do thực hiện pooling",
            "Có vòng lặp nội bộ tái diễn nhưng gặp vấn đề về độ dốc biến mất trên chuỗi dài",
            "Sử dụng thêm một luồng bộ nhớ nhằm duy trì trạng thái qua nhiều dấu thời gian"
        ],
        type: "matching",
        difficulty: "Trung bình"
    },
    {
        question: "Sắp xếp các bước thực hiện quá trình chuẩn bị cấu trúc dữ liệu chuỗi thời gian nhằm đưa vào huấn luyện mô hình dự báo:",
        options: [
            "Đọc dữ liệu từ file văn bản và tiến hành tách thành các mảng NumPy riêng biệt",
            "Tính giá trị trung bình và độ lệch chuẩn ưu tiên trên phần dữ liệu huấn luyện",
            "Chuẩn hóa dữ liệu bằng cách trừ đi trung bình và chia đều cho độ lệch chuẩn",
            "Sử dụng tiện ích timeseries_dataset_from_array để tạo các mẫu đầu vào batch"
        ],
        answer: [0, 1, 2, 3],
        type: "sorting",
        difficulty: "Khó"
    },
    {
        question: "Điền từ thích hợp vào chỗ trống: \nMạng RNN _____ khai thác độ nhạy thứ tự bằng cách xử lý trình tự đầu vào theo cả chiều xuôi và chiều ngược rồi hợp nhất các biểu diễn.",
        options: [
            "hai chiều"
        ],
        type: "fill",
        difficulty: "Dễ"
    },
    {
        question: "Điền từ thích hợp vào chỗ trống: \nChỉ số đánh giá độ chính xác được dùng cho phương pháp cơ sở thông thường trong bài toán dự báo nhiệt độ của chương 13 là _____ (chỉ cần viết viết tắt 3 chữ cái tiếng Anh).",
        options: [
            "MAE",
            "mae"
        ],
        type: "fill",
        difficulty: "Trung bình"
    }
];

export default questions;
