const questions = [
    {
        type: "mcq",
        question: "Theo chương 19, đặc điểm nào sau đây KHÔNG phải là giới hạn của các mô hình học sâu hiện tại?",
        options: [
            "Chúng là các cơ sở dữ liệu tĩnh.",
            "Chúng kém trong khả năng thích ứng với sự mới lạ.",
            "Chúng có khả năng thực hiện sự trừu tượng hóa lấy chương trình làm trung tâm (program-centric).",
            "Chúng rất nhạy cảm với cách diễn đạt đầu vào."
        ],
        answer: 2,
        explanation: "Các mô hình học sâu rất giỏi trong sự trừu tượng lấy giá trị làm trung tâm (value-centric) nhưng về cơ bản không có khả năng tạo ra sự trừu tượng lấy chương trình làm trung tâm (program-centric, như cấu trúc logic, lý luận rời rạc).",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Khi đối mặt với các vấn đề hoàn toàn mới lạ dù rất đơn giản (ví dụ như bài toán giải đố ARC), các mô hình ngôn ngữ lớn (LLM) thường thất bại vì lý do gì?",
        options: [
            "Khả năng giải quyết phụ thuộc vào sự quen thuộc với dữ liệu huấn luyện chứ không phải độ phức tạp.",
            "Chúng không có đủ số lượng tham số để tính toán logic cho các nhiệm vụ mang tính chất hình học.",
            "Bộ nhớ cục bộ của chúng không đủ dung lượng để lưu trữ ngữ cảnh dài của các câu đố mới.",
            "Các bộ lọc Convolutional không thể nắm bắt được chi tiết của bài toán trong không gian tiềm ẩn."
        ],
        answer: 0,
        explanation: "LLM giải quyết vấn đề bằng cách nội suy từ các mẫu quen thuộc đã thấy trong quá trình đào tạo. Nếu bài toán hoàn toàn mới và không có tương đương trực tiếp trong tập dữ liệu, chúng sẽ bó tay dù bài toán có đơn giản đến đâu.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Kỹ thuật Prompt Engineering (Kỹ sư Prompt) thực chất là gì theo góc nhìn của tác giả?",
        options: [
            "Một quá trình dạy cho AI những khái niệm mới chưa từng có trước đây.",
            "Một quá trình tìm kiếm thông qua không gian tiềm ẩn để tìm truy vấn tra cứu tối ưu nhất.",
            "Một quá trình tái lập trình cấu trúc Transformer để hiểu ý định của con người.",
            "Một quá trình nén kiến thức ngôn ngữ tự nhiên thành các tham số tĩnh."
        ],
        answer: 1,
        explanation: "Prompt engineering về bản chất là quá trình thử và sai để dò tìm một vị trí (địa chỉ) trong không gian tiềm ẩn rộng lớn chứa các chương trình/kiến thức mà mô hình đã học được, sao cho truy vấn đó khớp với nhiệm vụ mục tiêu.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Sự phân loại ngay lập tức, trực giác và nhận dạng mẫu dựa vào loại trừu tượng lấy ________ làm trung tâm.",
        answer: "giá trị",
        explanation: "Sự trừu tượng lấy giá trị làm trung tâm (Value-centric abstraction) dựa trên sự tương đồng liên tục, tạo ra các nguyên mẫu để nhận dạng khuôn mẫu và trực giác. Học sâu xuất sắc ở điểm này.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Trong khi các hệ thống trí tuệ nhân tạo trước đây bị giới hạn ở 'khái quát hóa cục bộ', trí tuệ của con người sở hữu khả năng ________ cực độ, cho phép thích ứng với những tình huống hoàn toàn mới lạ.",
        answer: "khái quát hóa",
        explanation: "Khái quát hóa cực độ (Extreme generalization) là khả năng thích ứng với những tình huống mới lạ, chưa từng có trước đây bằng cách sử dụng ít hoặc không có dữ liệu mới, điều mà con người làm được nhưng AI hiện tại thì không.",
        difficulty: "Trung bình"
    },
    {
        type: "matching",
        question: "Ghép nối các thuật ngữ khái quát hóa với đặc điểm tương ứng của chúng:",
        pairs: [
            { left: "Khái quát hóa cục bộ (Local generalization)", right: "Chỉ xử lý được các tình huống tương tự với dữ liệu huấn luyện (ẩn số đã biết)." },
            { left: "Khái quát hóa rộng rãi (Broad generalization)", right: "Xử lý được các tình huống chưa biết trong một lĩnh vực rộng lớn (VD: Xe tự lái)." },
            { left: "Khái quát hóa cực độ (Extreme generalization)", right: "Thích ứng nhanh chóng với tình huống hoàn toàn mới lạ bằng khả năng trừu tượng hóa." }
        ],
        explanation: "Học sâu hiện tại dừng ở mức Khái quát hóa cục bộ. AI tự hành đang hướng đến Khái quát hóa rộng rãi, trong khi con người sở hữu Khái quát hóa cực độ.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Trong quá trình tiến hóa, bộ não ban đầu hoạt động như các 'máy tự động' (automatons). Điểm khác biệt lớn nhất giữa một máy tự động và một tác nhân thông minh là gì?",
        options: [
            "Máy tự động có dung lượng bộ nhớ lớn hơn nhiều so với tác nhân thông minh.",
            "Máy tự động xử lý tốt dữ liệu phi cấu trúc, còn tác nhân thông minh thì không.",
            "Máy tự động tĩnh và chỉ phản ứng theo 'Nếu - Thì', còn tác nhân thông minh có thể thích ứng.",
            "Máy tự động học hỏi thông qua giảm độ dốc, trong khi tác nhân thông minh dùng thuật toán gen."
        ],
        answer: 2,
        explanation: "Máy tự động thực thi các chương trình hành vi được lập trình sẵn (cứng nhắc), trong khi các tác nhân thông minh có thể linh hoạt điều chỉnh và tạo ra mô hình trừu tượng để thích ứng với tình huống bất ngờ.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Theo 'Giả thuyết kính vạn hoa' (Kaleidoscope hypothesis), trí thông minh xử lý sự mới lạ bằng cách nào?",
        options: [
            "Xóa bỏ những ký ức cũ để tạo không gian lưu trữ trực tiếp cho các tình huống mới.",
            "Nhận ra rằng mọi thứ mới đều là sự kết hợp lại của các 'nguyên tử ý nghĩa' đã biết.",
            "Huấn luyện lại toàn bộ mạng lưới nơ-ron mỗi khi gặp phải một trường hợp chưa có dữ liệu.",
            "Thực hiện việc tìm kiếm ngẫu nhiên trên toàn bộ không gian hành vi cho đến khi thành công."
        ],
        answer: 1,
        explanation: "Giả thuyết kính vạn hoa cho rằng sự mới lạ vô tận của thế giới thực ra chỉ là sự tái kết hợp của một số lượng nhỏ các 'nguyên tử ý nghĩa' (sự trừu tượng cốt lõi) giống như các mảnh kính nhỏ trong kính vạn hoa.",
        difficulty: "Trung bình"
    },
    {
        type: "sorting",
        question: "Sắp xếp quá trình giải quyết vấn đề mới của một trí tuệ thông minh (theo giả thuyết kính vạn hoa):",
        order: [
            "Trải nghiệm sự vật hiện tượng và trích xuất các 'nguyên tử ý nghĩa' cốt lõi (Abstraction acquisition).",
            "Lưu trữ các khối trừu tượng này thành một bộ sưu tập.",
            "Gặp một tình huống hoàn toàn mới mẻ chưa từng đối mặt trước đây.",
            "Kết hợp lại nhanh chóng các khối trừu tượng (On-the-fly recombination) để tạo ra mô hình xử lý phù hợp."
        ],
        explanation: "Bản chất của trí thông minh gồm 2 bước chính: Thu thập sự trừu tượng từ quá khứ, và Kết hợp lại chúng một cách nhanh chóng khi đối mặt với tình huống mới.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Ví dụ nào sau đây thể hiện hiện tượng 'ví dụ đối nghịch' (adversarial example) ở các mô hình thị giác máy tính?",
        options: [
            "Mô hình không nhận diện được ảnh bị làm mờ một phần do nhiễu tự nhiên.",
            "Thay đổi một vài pixel nhỏ không thể nhìn thấy bằng mắt thường khiến ảnh con gấu trúc bị nhận diện thành con vượn.",
            "Mô hình nhận diện sai một chiếc xe tải vì góc chụp từ trên cao xuống chưa từng có trong tập huấn luyện.",
            "Ảnh có độ phân giải thấp được mô hình upscale thành ảnh độ nét cao nhưng mất đi một số chi tiết thật."
        ],
        answer: 1,
        explanation: "Các ví dụ đối nghịch là các tinh chỉnh có chủ ý, cực nhỏ (gradient ascent trong không gian đầu vào) để đánh lừa mô hình, chứng minh rằng cách mô hình hiểu hình ảnh rất khác so với con người.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tại sao tỷ lệ giải bài toán cộng hai số lớn (ví dụ: 4357 + 8936) của mô hình Transformer lại không đạt 100% dù được học hàng triệu ví dụ?",
        options: [
            "Vì giới hạn phần cứng không cho phép lưu trữ đủ số lượng các chuỗi độ dài lớn.",
            "Vì nó cố gắng nội suy biến đổi hình học thay vì học các bước logic rời rạc (carrying over).",
            "Vì hàm kích hoạt softmax không được thiết kế cho việc tạo ra giá trị số học tuyệt đối.",
            "Vì dữ liệu huấn luyện thường chứa các nhãn bị gán sai do lỗi sinh tự động từ con người."
        ],
        answer: 1,
        explanation: "Học sâu ánh xạ dữ liệu qua các biến đổi hình học liên tục. Nó rất kém trong việc học và áp dụng các logic rời rạc từng bước (như phép nhớ trong phép cộng), nên nó chỉ đoán bằng cách nội suy thay vì thực hiện thuật toán thực sự.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Mục đích chính của bộ chuẩn ARC-AGI do François Chollet tạo ra là gì?",
        options: [
            "Đánh giá xem một LLM có thể lưu trữ bao nhiêu lượng dữ liệu văn bản từ internet.",
            "Đo lường trí thông minh bằng cách kiểm tra hệ thống trên các nhiệm vụ không thể đoán trước được.",
            "Đánh giá kỹ năng lập trình của con người khi thi đấu cùng với các công cụ Copilot.",
            "Đo lường thời gian cần thiết để huấn luyện một mô hình học sâu trên tập dữ liệu hoàn toàn mới."
        ],
        answer: 1,
        explanation: "ARC-AGI kiểm tra hệ thống AI đối với những nhiệm vụ hoàn toàn mới, chưa được báo trước để xem hệ thống có thực sự sở hữu khả năng thích ứng linh hoạt (trí thông minh) thay vì chỉ sử dụng sự quen thuộc từ ghi nhớ.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Sự phân chia 'trừu tượng lấy giá trị làm trung tâm' và 'trừu tượng lấy chương trình làm trung tâm' có sự tương đồng với hệ thống nào trong tâm lý học nhận thức của con người?",
        options: [
            "Trí nhớ ngắn hạn và Trí nhớ dài hạn.",
            "Hệ thống 1 (Trực giác, nhanh, tự động) và Hệ thống 2 (Lý luận, chậm, có chủ ý).",
            "Bán cầu não trái và Bán cầu não phải.",
            "Phản xạ có điều kiện và Phản xạ không điều kiện."
        ],
        answer: 1,
        explanation: "Trừu tượng giá trị đại diện cho nhận thức, trực giác nhanh (Hệ thống 1). Trừu tượng chương trình đại diện cho sự lý luận, tính toán, lập kế hoạch có chủ ý và chậm rãi hơn (Hệ thống 2).",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "TTA là viết tắt của Test-Time ________, một cơ chế cho phép AI thực hiện suy luận hoặc học tập tích cực trong quá trình kiểm tra dựa trên thông tin bài toán.",
        answer: "Adaptation",
        explanation: "Test-Time Adaptation (Thích ứng thời gian thử nghiệm) là chìa khóa mà các mô hình như o3 sử dụng để đạt hiệu suất cao trên ARC-AGI bằng cách tìm kiếm/huấn luyện tại thời điểm inference.",
        difficulty: "Trung bình"
    },
    {
        type: "matching",
        question: "Ghép nối phương pháp TTA với nguyên lý hoạt động của nó:",
        pairs: [
            { left: "Huấn luyện trong thời gian thử nghiệm", right: "Sử dụng giảm độ dốc để điều chỉnh tham số mô hình ngay trên ví dụ của bài kiểm tra." },
            { left: "Phương pháp tìm kiếm ngôn ngữ tự nhiên", right: "Tạo ra và đánh giá nhiều chuỗi suy nghĩ (Chain of Thought) để tìm ra câu trả lời hợp lý." },
            { left: "Tổng hợp chương trình (Program synthesis)", right: "Tự động tạo ra mã nguồn (chương trình rời rạc) để khớp với các đầu vào-đầu ra được cung cấp." }
        ],
        explanation: "Đây là các phương pháp tiếp cận chính của Test-Time Adaptation giúp mô hình xử lý tính mới mà không cần phụ thuộc hoàn toàn vào dữ liệu huấn luyện trước đó.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Hiệu ứng 'Quy tắc tắt' (Shortcut rule) trong AI được tác giả miêu tả có ý nghĩa gì?",
        options: [
            "Các mô hình sẽ luôn tìm được con đường ngắn nhất trong đồ thị tìm kiếm nhờ thuật toán A*.",
            "Việc tối ưu hóa một chỉ số thành công sẽ dẫn đến mô hình sử dụng mọi mánh khóe để đạt điểm mà không cần thông minh.",
            "Khi tham số quá lớn, việc tính toán có thể được rút ngắn bằng lượng tử hóa để suy luận nhanh hơn.",
            "Việc kết hợp nhiều mô hình nhỏ lại sẽ tạo ra một đường tắt để vượt qua giới hạn của học sâu."
        ],
        answer: 1,
        explanation: "Khi mục tiêu chỉ là 'giải quyết nhiệm vụ X', hệ thống hoặc người nghiên cứu sẽ tìm mọi lối tắt (ghi nhớ, mã hóa cứng, dùng thêm dữ liệu) để đạt điểm cao mà không cần đến sự thông minh thực sự.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Theo tác giả, vì sao Deep Blue (AI chơi cờ của IBM) lại KHÔNG phản ánh trí tuệ nhân tạo tổng quát?",
        options: [
            "Vì thuật toán A-star của nó chỉ là thuật toán tìm kiếm cục bộ không thể tính toán sâu.",
            "Vì năng lực tính toán của thập niên 90 chưa đủ để nó chạy các mạng lưới thần kinh lớn.",
            "Vì chơi cờ bằng tìm kiếm không cần đến các mô hình trừu tượng linh hoạt có thể khái quát sang nhiệm vụ khác.",
            "Vì nó cần sự can thiệp của con người giữa các ván đấu để điều chỉnh lại cấu hình."
        ],
        answer: 2,
        explanation: "Deep Blue chỉ dùng sức mạnh tính toán để duyệt không gian trạng thái cờ vua cực tốt. Nó giải quyết triệt để 1 nhiệm vụ cố định nhưng không có một chút khả năng khái quát hóa sang bất kỳ lĩnh vực nào khác.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Điều nào mô tả đúng nhất sự khác biệt giữa ARC-AGI-1 và ARC-AGI-2?",
        options: [
            "ARC-AGI-2 cho phép sử dụng hình ảnh 3D thay vì chỉ các lưới 2D đơn giản.",
            "ARC-AGI-2 yêu cầu các hệ thống phải giải quyết bằng lượng điện toán nhỏ đi đáng kể.",
            "ARC-AGI-2 đòi hỏi chuỗi suy luận phức tạp hơn và chống lại các phương pháp duyệt toàn diện (brute-force) tốt hơn.",
            "ARC-AGI-2 bao gồm các câu đố ngôn ngữ tự nhiên để thử thách các mô hình ngôn ngữ lớn hiện tại."
        ],
        answer: 2,
        explanation: "ARC-AGI-2 nâng cao độ khó để các phương pháp dựa trên sức mạnh tính toán thuần túy (brute-force search) của o3 gặp khó khăn, buộc các hệ thống phải sử dụng chiến lược giải quyết hiệu quả hơn.",
        difficulty: "Trung bình"
    },
    {
        type: "sorting",
        question: "Sắp xếp theo thứ tự tiến hóa của sự trừu tượng/trí thông minh (từ thấp đến cao) theo góc nhìn trong chương:",
        order: [
            "Hệ thống tự động hóa (Automatons) thực thi các chương trình hành vi được mã hóa sẵn.",
            "Khái quát hóa cục bộ (Local generalization) xử lý các biến thể nhỏ của tình huống quen thuộc.",
            "Khái quát hóa rộng rãi (Broad generalization) xử lý các tình huống bất ngờ trong một lĩnh vực rộng.",
            "Khái quát hóa cực độ (Extreme generalization) sử dụng trừu tượng để thích ứng ngay lập tức với sự mới lạ hoàn toàn."
        ],
        explanation: "Từ côn trùng, đến mô hình học sâu hiện tại, đến AI tự hành tương lai, và cuối cùng là trí tuệ của con người hiện đại.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Trong phương pháp tổng hợp chương trình (Program synthesis), việc tìm kiếm chương trình phù hợp thường được định hướng bằng cách nào?",
        options: [
            "Đánh giá thông qua việc tính toán độ dốc (gradient descent) trực tiếp trên các dòng mã.",
            "Chạy thử chương trình và kiểm tra xem kết quả đầu ra có khớp với các ví dụ được cung cấp hay không.",
            "So sánh tính tương đồng về mặt thị giác của biểu đồ luồng chương trình với dữ liệu.",
            "Phân loại các đoạn mã thông qua một mạng nơ-ron Transformer đồ thị khổng lồ."
        ],
        answer: 1,
        explanation: "Chương trình (mã nguồn rời rạc) không thể tính đạo hàm. Việc tìm kiếm được thực hiện bằng cách khởi tạo nhiều chương trình, thực thi chúng và kiểm tra độ chính xác trên các tập dữ liệu đầu vào-đầu ra mong muốn.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Đâu là một nhược điểm chính của việc sử dụng 'Lực đẩy không gian giải pháp bằng lượng điện toán khổng lồ' (như cách o3 giải ARC-AGI)?",
        options: [
            "Nó có thể dễ dàng bị giới hạn bởi các lỗi tràn bộ nhớ trên các GPU tiêu chuẩn.",
            "Nó chỉ giải quyết được các câu hỏi dạng phân loại thay vì dạng sinh văn bản hoặc tạo lưới.",
            "Nó không phản ánh trí thông minh thực sự vì tính hiệu quả chuyển đổi từ kinh nghiệm sang kỹ năng là rất thấp.",
            "Nó đòi hỏi phải có một kỹ sư điều chỉnh siêu tham số trực tiếp trong suốt quá trình chạy."
        ],
        answer: 2,
        explanation: "Bản chất của trí thông minh là ở TÍNH HIỆU QUẢ. Việc duyệt qua hàng tỷ khả năng bằng tài nguyên khổng lồ (hàng chục ngàn đô la) để giải một câu đố là cách dùng sức mạnh cơ bắp thay vì trí thông minh linh hoạt.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Trong 'Trừu tượng lấy chương trình làm trung tâm', chúng ta không so sánh bằng khoảng cách liên tục, mà bằng cách tìm kiếm sự khớp ________ chính xác (ví dụ: đẳng cấu đồ thị con).",
        answer: "cấu trúc",
        explanation: "Khớp cấu trúc (structural matching) hoặc đẳng cấu đồ thị con (subgraph isomorphism) là cốt lõi của lý luận logic, giúp tái sử dụng các khuôn mẫu thuật toán chính xác thay vì chỉ tương đồng về mặt hình ảnh.",
        difficulty: "Trung bình"
    },
    {
        type: "fill",
        question: "Sự có mặt của *lý thuyết tâm trí* (________ of mind) khiến con người dễ lầm tưởng rằng mô hình LLM hiểu ngôn ngữ hệt như chúng ta, khi thực ra chúng chỉ đang khớp chuỗi thống kê.",
        answer: "theory",
        explanation: "Theory of mind khiến con người tự động gán ý định, sự hiểu biết và cảm xúc cho những thực thể biểu hiện hành vi phức tạp (như máy sinh văn bản).",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Theo tác giả, vì sao các LLM lại thất bại khi giải câu đố 'Alice có N anh trai và M chị gái...' nếu thay đổi M và N khác với các giá trị thông thường trên mạng?",
        options: [
            "Vì các con số lớn hơn làm tràn dung lượng biểu diễn dấu phẩy động của mô hình.",
            "Vì mô hình không hiểu cấu trúc logic cốt lõi mà chỉ truy xuất câu trả lời đã ghi nhớ dựa vào cụm từ quen thuộc.",
            "Vì mô hình dịch sai ngữ pháp câu hỏi sang biểu diễn không gian vectơ đa chiều.",
            "Vì mô hình cho rằng Alice không phải là một thành viên trong chính gia đình đó."
        ],
        answer: 1,
        explanation: "Mô hình nhận dạng được mẫu văn bản giống với câu đố quen thuộc và ngay lập tức gọi ra câu trả lời đã học thuộc lòng mà không áp dụng một phép tính logic rời rạc nào với M và N mới.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tỷ lệ hiệu quả (Efficiency ratio) - thước đo của trí thông minh theo Chollet - là tỷ lệ chuyển đổi giữa hai yếu tố nào?",
        options: [
            "Kích thước của tập dữ liệu huấn luyện và Số lượng tham số của mô hình mạng.",
            "Số lượng GPU sử dụng và Thời gian cần thiết để mô hình hoàn thành quá trình fine-tuning.",
            "Lượng kinh nghiệm/thông tin liên quan đã có và Khu vực hoạt động trong tương lai (tập hợp các tình huống mới có thể xử lý).",
            "Độ chính xác trên tập kiểm định và Lượng điện toán tiêu thụ trong thời gian kiểm tra."
        ],
        answer: 2,
        explanation: "Trí thông minh đo lường việc bạn có thể tận dụng một lượng nhỏ kinh nghiệm/thông tin quá khứ để thích ứng với một dải rộng lớn các nhiệm vụ và tình huống mới lạ trong tương lai tốt đến mức nào.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Việc sử dụng các bộ 'kiến thức cốt lõi' (Core Knowledge) trong ARC-AGI nhằm mục đích gì?",
        options: [
            "Để ưu tiên những thí sinh có vốn từ vựng tiếng Anh phong phú.",
            "Để tránh sự thiên lệch và đảm bảo rằng nhiệm vụ không yêu cầu kiến thức đặc thù thu thập từ văn hóa con người.",
            "Để mô hình chỉ phải giải quyết các bài toán đại số tuyến tính cao cấp.",
            "Để đánh lừa mô hình LLM bằng cách cung cấp các gợi ý sai lệch vào prompt."
        ],
        answer: 1,
        explanation: "Core Knowledge đại diện cho các khái niệm bẩm sinh (như đối tượng vật lý, số lượng cơ bản, không gian) mà bất kỳ con người nào cũng có, đảm bảo tính công bằng: khả năng suy luận chứ không phải khối lượng kiến thức học vẹt được đo lường.",
        difficulty: "Trung bình"
    },
    {
        type: "matching",
        question: "Phân loại các phương pháp/thuật toán vào loại trừu tượng phù hợp:",
        pairs: [
            { left: "K-means clustering", right: "Trừu tượng lấy giá trị làm trung tâm (Value-centric)" },
            { left: "Convolutional Neural Networks (Nhận dạng ảnh)", right: "Trừu tượng lấy giá trị làm trung tâm (Value-centric)" },
            { left: "Kế thừa Class trong Lập trình Hướng đối tượng", right: "Trừu tượng lấy chương trình làm trung tâm (Program-centric)" },
            { left: "Tổng hợp mã nguồn (Program Synthesis)", right: "Trừu tượng lấy chương trình làm trung tâm (Program-centric)" }
        ],
        explanation: "Các thuật toán học máy phần lớn dựa trên khoảng cách liên tục (value-centric). Kỹ thuật phần mềm và logic dựa trên cấu trúc rời rạc (program-centric).",
        difficulty: "Trung bình"
    },
    {
        type: "sorting",
        question: "Quy trình lý tưởng của phương pháp 'Huấn luyện trong thời gian thử nghiệm' (Test-Time Training) trên một nhiệm vụ ARC-AGI:",
        order: [
            "Nhận một nhiệm vụ mới với một vài cặp ví dụ (Đầu vào - Đầu ra).",
            "Cập nhật trực tiếp các tham số của mô hình thông qua gradient descent ngay trên các cặp ví dụ này.",
            "Nhận lưới đầu vào thử nghiệm cuối cùng của nhiệm vụ đó.",
            "Sử dụng mô hình vừa được tinh chỉnh tức thời để đưa ra dự đoán lưới đầu ra."
        ],
        explanation: "Thay vì chỉ infer tĩnh, Test-Time Training tận dụng ngay chính bài kiểm tra để tự điều chỉnh cục bộ, cho phép mô hình thích ứng ngay tại chỗ với quy luật của bài tập mới.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Tác giả kết luận điều gì về 'Quy mô là tất cả những gì bạn cần' (Scale is all you need)?",
        options: [
            "Đó là chân lý tuyệt đối sẽ dẫn tới AGI trong vài năm tới nếu tiếp tục tăng quy mô dữ liệu.",
            "Quy mô lớn chỉ làm tăng khả năng ghi nhớ và khái quát hóa cục bộ, không tạo ra trí thông minh linh hoạt.",
            "Quy mô chỉ cần thiết đối với mô hình ngôn ngữ còn đối với thị giác máy tính thì hoàn toàn vô dụng.",
            "Quy mô làm giảm đi khả năng hiểu biết ngữ pháp của mô hình ngôn ngữ qua từng thế hệ."
        ],
        answer: 1,
        explanation: "Dù mô hình lớn đến đâu, chúng vẫn dùng cơ chế nội suy trên đa tạp dữ liệu tĩnh. Việc tăng quy mô không giải quyết được vấn đề cơ bản: khả năng thích ứng với điều chưa biết bằng trí thông minh thực sự.",
        difficulty: "Trung bình"
    },
    {
        type: "mcq",
        question: "Điều nào mô tả chính xác nhất lý do tại sao AI thông minh (AGI) vẫn là một 'cánh đồng xanh' (greenfield)?",
        options: [
            "Vì phần cứng hiện tại chưa đủ mạnh để huấn luyện mạng với hàng ngàn tỷ tham số.",
            "Vì giới nghiên cứu vẫn đang thiếu những bộ dữ liệu nhãn khổng lồ chất lượng cao.",
            "Vì học sâu thực chất chỉ là tự động hóa nhận thức, trong khi sự thông minh thật sự đòi hỏi sự kết hợp của lý luận và tái kết hợp cấu trúc.",
            "Vì các công ty công nghệ lớn không muốn chia sẻ các công nghệ mã nguồn mở của họ."
        ],
        answer: 2,
        explanation: "Ngành AI hiện tại thành công rực rỡ ở mảng 'tự động hóa nhận thức' (cognitive automation). Nhưng mảng 'trí tuệ nhân tạo' (sự hiểu biết, khả năng thích ứng linh hoạt) thực sự mới chỉ ở bước khởi đầu.",
        difficulty: "Trung bình"
    }
];

export default questions;

